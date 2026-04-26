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
            // Start a new sentence
            let snap = captureSignals?() ?? SpeechSegment.SignalSnapshot()
            activeSentence = Sentence(
                text: text,
                isFinal: false,
                startedLookingAtScreen: isLookingAtScreen,
                gazeSpans: isLookingAtScreen ? [GazeSpan(charCount: newCharCount, isToScreen: true)] : [],
                isFromUser: isSpeaking,
                signals: snap
            )
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else {
            // Update existing sentence
            activeSentence?.text = text

            // isFromUser and signals are locked at sentence creation — don't update mid-sentence

            if addedChars > 0, activeSentence?.startedLookingAtScreen == true {
                updateGazeSpans(addedChars: addedChars)
            } else if newCharCount != lastCharCount, activeSentence?.startedLookingAtScreen == true {
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

    /// Exponentially-weighted fraction of characters spoken while looking at the screen.
    ///
    /// For each character `i` in the sentence, weight `w_i = exp(-λ * i)`.
    /// Score = Σ(w_i * isToScreen_i) / Σ(w_i), clamped to [0, 1].
    /// Empty text → 0. Result is `1.0` when every weighted char is on-screen,
    /// `0.0` when none are, and something in-between when gaze shifts.
    private func speakingToAIScore(for s: Sentence) -> Float {
        let totalChars = s.gazeSpans.reduce(0) { $0 + $1.charCount }
        guard totalChars > 0 else {
            // No gaze spans recorded — fall back to the bool for a sane default.
            return s.startedLookingAtScreen ? 1.0 : 0.0
        }

        var weightedOn: Float = 0
        var weightedTotal: Float = 0
        var index: Int = 0
        for span in s.gazeSpans {
            for _ in 0..<span.charCount {
                let w = expf(-gazeDecayLambda * Float(index))
                weightedTotal += w
                if span.isToScreen { weightedOn += w }
                index += 1
            }
        }
        guard weightedTotal > 0 else { return 0 }
        return max(0, min(1, weightedOn / weightedTotal))
    }

    // MARK: - Private: Gaze Helpers

    private func updateGazeSpans(addedChars: Int) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        let lastSpan = spans[spans.count - 1]
        if lastSpan.isToScreen == isLookingAtScreen {
            spans[spans.count - 1] = GazeSpan(charCount: lastSpan.charCount + addedChars, isToScreen: lastSpan.isToScreen)
        } else {
            spans.append(GazeSpan(charCount: addedChars, isToScreen: isLookingAtScreen))
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
