#if os(iOS)
import Foundation

/// Layer 2: Sentence building with SpeechAnalyzer's volatile/final model.
///
/// Volatile results → display text (may change)
/// Final results → confirmed text (never changes)
///
/// No more manual silence splitting — Apple handles the volatile→final
/// transition. We just track sentences and build segments.
@MainActor
public class SentenceBuilder {

    // MARK: - Types

    private struct GazeSpan {
        let id = UUID()
        var charCount: Int
        let isToScreen: Bool
    }

    private struct Sentence {
        let id = UUID()
        var text: String
        var isFinal: Bool
        var startedLookingAtScreen: Bool
        var gazeSpans: [GazeSpan]
        var isFromUser: Bool
        var signals: SpeechSegment.SignalSnapshot
        /// Snapshot of the onset-weighted gaze score at sentence creation.
        /// Computed by HumanStateEngine from audio-frame-rate samples over
        /// the first 500ms of voice activity (exponential decay). Fixed for
        /// the lifetime of the sentence — mid-sentence glances don't shift
        /// the verdict, matching user intent: "the start of the utterance
        /// defines whether the user was addressing the screen."
        var onsetGazeScore: Float
    }

    // MARK: - State

    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    private var lastCharCount: Int = 0
    private var speechStartCaptured: Bool = false

    /// Exponential decay rate for "talking-to-AI" scoring.
    /// λ=0.5: char 0 weight=1.0, char 3 ≈20%, char 5 ≈8%, char 10 ≈0.7%.
    /// The first 3-5 characters dominate — matches the intuition that users
    /// turn their gaze to the screen near the start of an utterance.
    private let gazeDecayLambda: Float = 0.5

    // External inputs
    var isLookingAtScreen: Bool = false
    var isSpeaking: Bool = false
    /// Live onset-weighted gaze score, updated at audio-frame rate by
    /// HumanStateEngine. Captured into each Sentence at creation time.
    var onsetGazeScore: Float = 0
    /// Closure to capture current signal snapshot for debug display
    var captureSignals: (() -> SpeechSegment.SignalSnapshot)?

    // MARK: - Public API

    /// Process a transcription result from SpeechAnalyzer.
    /// `isFinal` means Apple has finalized this segment — text won't change.
    func handleResult(text: String, isFinal: Bool) {
        let newCharCount = text.count
        let addedChars = max(0, newCharCount - lastCharCount)

        print("[SentenceBuilder] handleResult: '\(text.prefix(40))' isFinal=\(isFinal) chars=\(newCharCount) isSpeaking=\(isSpeaking)")

        if activeSentence == nil {
            // Start a new sentence. Always seed a gaze span covering the
            // current text so the score sees *every* character, regardless of
            // whether the user started by looking at the screen.
            let snap = captureSignals?() ?? SpeechSegment.SignalSnapshot()
            let seedSpan = newCharCount > 0
                ? [GazeSpan(charCount: newCharCount, isToScreen: isLookingAtScreen)]
                : []
            activeSentence = Sentence(
                text: text,
                isFinal: false,
                startedLookingAtScreen: isLookingAtScreen,
                gazeSpans: seedSpan,
                isFromUser: isSpeaking,
                signals: snap,
                onsetGazeScore: onsetGazeScore
            )
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else {
            // Update existing sentence
            activeSentence?.text = text

            // isFromUser is — as of 4.9.8 — decided from the *speech onset*
            // window (the first ~500ms of voice activity, sampled with
            // exponential decay on the audio-frame timeline). The
            // SpeechOnsetTracker runs inside HumanStateEngine and latches
            // isSpeaking at the right moment, so by the time STT emits its
            // first volatile text, isSpeaking already reflects the onset
            // verdict. We must NOT upgrade mid-sentence based on later
            // lipCorrelated blips — glancing at the screen while someone
            // else speaks would otherwise flip the sentence to user.
            //
            // So: isFromUser and signals are captured once at sentence
            // creation and stay locked, matching the user's preferred
            // semantics ("first few chars decide").

            // Always update gaze spans so the score reflects the entire
            // sentence — not just sentences that started on-screen.
            if addedChars > 0 {
                updateGazeSpans(addedChars: addedChars)
            } else if newCharCount != lastCharCount {
                ensureGazeSpansCoverText(text)
            }
            lastCharCount = newCharCount
        }

        if isFinal {
            // Apple says this segment is done — finalize it
            activeSentence?.isFinal = true
            finalizeActiveSentence()
        }
    }

    /// Finalize the active sentence and prepare for a new one.
    func finalizeAndReset() {
        finalizeActiveSentence()
        resetActive()
    }

    /// Reset active sentence state.
    func resetActive() {
        activeSentence = nil
        lastCharCount = 0
        speechStartCaptured = false
    }

    /// Clear all sentences and segments.
    func clearAll() {
        sentences = []
        activeSentence = nil
    }

    /// Build the current segments array from sentences + active sentence.
    func buildSegments() -> [SpeechSegment] {
        var result: [SpeechSegment] = []

        for s in sentences { appendSentence(s, to: &result) }
        if let active = activeSentence, !active.text.isEmpty {
            appendSentence(active, to: &result)
        }

        return result
    }

    // MARK: - Private: Sentence Lifecycle

    private func finalizeActiveSentence() {
        guard let active = activeSentence, !active.text.isEmpty else {
            activeSentence = nil
            return
        }
        sentences.append(active)
        if sentences.count > maxSentences { sentences.removeFirst() }
        activeSentence = nil
    }

    // MARK: - Private: Segment Building

    private func appendSentence(_ s: Sentence, to result: inout [SpeechSegment]) {
        let score = speakingToAIScore(for: s)

        if !result.isEmpty {
            result.append(SpeechSegment(
                text: " ", isToScreen: false,
                sentenceStartedLookingAtScreen: s.startedLookingAtScreen,
                speakingToAIScore: score,
                isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals
            ))
        }

        if !s.startedLookingAtScreen {
            result.append(SpeechSegment(
                id: s.id, text: s.text, isToScreen: false,
                sentenceStartedLookingAtScreen: false,
                speakingToAIScore: score,
                isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals
            ))
        } else {
            var offset = s.text.startIndex
            for (i, span) in s.gazeSpans.enumerated() {
                let end = s.text.index(offset, offsetBy: span.charCount, limitedBy: s.text.endIndex) ?? s.text.endIndex
                let spanText = String(s.text[offset..<end])
                if !spanText.isEmpty {
                    result.append(SpeechSegment(
                        id: i == 0 ? s.id : span.id,
                        text: spanText, isToScreen: span.isToScreen,
                        sentenceStartedLookingAtScreen: true,
                        speakingToAIScore: score,
                        isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals
                    ))
                }
                offset = end
            }
            if offset < s.text.endIndex {
                let remaining = String(s.text[offset...])
                if !remaining.isEmpty {
                    result.append(SpeechSegment(
                        text: remaining,
                        isToScreen: s.gazeSpans.last?.isToScreen ?? true,
                        sentenceStartedLookingAtScreen: true,
                        speakingToAIScore: score,
                        isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals
                    ))
                }
            }
        }
    }

    // MARK: - Private: Scoring

    /// Returns the onset-weighted gaze score captured at sentence creation.
    ///
    /// This score is computed by `HumanStateEngine` on the **audio-frame**
    /// timeline (60fps), sampling the first 500ms of voice activity with
    /// exponential decay (e-fold every 150ms). The per-character gaze span
    /// approach that used to live here could only sample at STT's volatile
    /// cadence (a few times per sentence) and therefore collapsed to 0/1
    /// whenever STT happened to emit a single big chunk. Audio-frame
    /// sampling avoids that aliasing.
    private func speakingToAIScore(for s: Sentence) -> Float {
        // Use per-char gaze spans (updated throughout the sentence) with
        // exponential decay weighting — early characters matter more.
        let spans = s.gazeSpans
        guard !spans.isEmpty else { return 0 }
        let totalChars = spans.reduce(0) { $0 + $1.charCount }
        guard totalChars > 0 else { return 0 }

        var score: Float = 0
        var totalWeight: Float = 0
        var charIndex = 0
        for span in spans {
            for _ in 0..<span.charCount {
                let t = Float(charIndex) / Float(max(1, totalChars - 1))
                let w = expf(-gazeDecayLambda * t)
                score += w * (span.isToScreen ? 1 : 0)
                totalWeight += w
                charIndex += 1
            }
        }
        return totalWeight > 0 ? score / totalWeight : 0
    }

    // MARK: - Private: Gaze Helpers

    private func updateGazeSpans(addedChars: Int) {
        var spans = activeSentence?.gazeSpans ?? []
        if spans.isEmpty {
            // Sentence started with no gaze span recorded (legacy callsite).
            // Seed one now so the new chars get tracked.
            spans.append(GazeSpan(charCount: addedChars, isToScreen: isLookingAtScreen))
        } else {
            let lastSpan = spans[spans.count - 1]
            if lastSpan.isToScreen == isLookingAtScreen {
                spans[spans.count - 1] = GazeSpan(charCount: lastSpan.charCount + addedChars, isToScreen: lastSpan.isToScreen)
            } else {
                spans.append(GazeSpan(charCount: addedChars, isToScreen: isLookingAtScreen))
            }
        }
        activeSentence?.gazeSpans = spans
    }

    private func ensureGazeSpansCoverText(_ text: String) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        let totalChars = spans.reduce(0) { $0 + $1.charCount }
        let textCount = text.count
        if totalChars != textCount {
            let lastIdx = spans.count - 1
            let newCount = max(0, spans[lastIdx].charCount + (textCount - totalChars))
            spans[lastIdx] = GazeSpan(charCount: newCount, isToScreen: spans[lastIdx].isToScreen)
            if spans[lastIdx].charCount == 0 && spans.count > 1 { spans.removeLast() }
            activeSentence?.gazeSpans = spans
        }
    }
}
#endif
