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
        var onsetGazeScore: Float
        var audioStartTime: Double?
        var audioEndTime: Double?
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
    func handleResult(text: String, isFinal: Bool, audioStartTime: Double? = nil, audioEndTime: Double? = nil) {
        let newCharCount = text.count
        let addedChars = max(0, newCharCount - lastCharCount)

        print("[SentenceBuilder] handleResult: '\(text.prefix(40))' isFinal=\(isFinal) chars=\(newCharCount) isSpeaking=\(isSpeaking)")

        if activeSentence == nil {
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
                onsetGazeScore: onsetGazeScore,
                audioStartTime: audioStartTime,
                audioEndTime: audioEndTime
            )
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else {
            activeSentence?.text = text
            if let end = audioEndTime { activeSentence?.audioEndTime = end }
            if activeSentence?.isFromUser == false && isSpeaking {
                activeSentence?.isFromUser = true
                if let snap = captureSignals?() {
                    activeSentence?.signals = snap
                }
            }
            if addedChars > 0 {
                updateGazeSpans(addedChars: addedChars)
            } else if newCharCount != lastCharCount {
                ensureGazeSpansCoverText(text)
            }
            lastCharCount = newCharCount
        }

        if isFinal {
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
                isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals,
                audioStartTime: s.audioStartTime, audioEndTime: s.audioEndTime
            ))
        }

        if !s.startedLookingAtScreen {
            result.append(SpeechSegment(
                id: s.id, text: s.text, isToScreen: false,
                sentenceStartedLookingAtScreen: false,
                speakingToAIScore: score,
                isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals,
                audioStartTime: s.audioStartTime, audioEndTime: s.audioEndTime
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
                        isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals,
                        audioStartTime: s.audioStartTime, audioEndTime: s.audioEndTime
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
                        isFromUser: s.isFromUser, isFinal: s.isFinal, signals: s.signals,
                        audioStartTime: s.audioStartTime, audioEndTime: s.audioEndTime
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
        // w = exp(-lambda * charIndex), so char 0 dominates and later chars
        // quickly decay to near-zero. This means the verdict is decided by
        // the first few chars of the sentence, not the tail.
        let spans = s.gazeSpans
        guard !spans.isEmpty else { return 0 }

        var score: Float = 0
        var totalWeight: Float = 0
        var charIndex = 0
        for span in spans {
            for _ in 0..<span.charCount {
                let w = expf(-gazeDecayLambda * Float(charIndex))
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
