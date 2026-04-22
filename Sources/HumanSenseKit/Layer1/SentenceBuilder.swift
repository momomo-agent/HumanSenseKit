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
    }

    // MARK: - State

    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    private var lastCharCount: Int = 0
    private var speechStartCaptured: Bool = false

    // External inputs
    var isLookingAtScreen: Bool = false
    var isSpeaking: Bool = false

    // MARK: - Public API

    /// Process a transcription result from SpeechAnalyzer.
    /// `isFinal` means Apple has finalized this segment — text won't change.
    func handleResult(text: String, isFinal: Bool) {
        let newCharCount = text.count
        let addedChars = max(0, newCharCount - lastCharCount)

        print("[SentenceBuilder] handleResult: '\(text.prefix(40))' isFinal=\(isFinal) chars=\(newCharCount)")

        if activeSentence == nil {
            // Start a new sentence
            activeSentence = Sentence(
                text: text,
                isFinal: false,
                startedLookingAtScreen: isLookingAtScreen,
                gazeSpans: isLookingAtScreen ? [GazeSpan(charCount: newCharCount, isToScreen: true)] : [],
                isFromUser: isSpeaking  // Only mark as user speech if lip-audio correlated
            )
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else {
            // Update existing sentence
            activeSentence?.text = text

            // If user starts speaking mid-sentence, upgrade to user speech
            if isSpeaking && activeSentence?.isFromUser == false {
                activeSentence?.isFromUser = true
            }

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
        if !result.isEmpty {
            result.append(SpeechSegment(
                text: " ", isToScreen: false,
                sentenceStartedLookingAtScreen: s.startedLookingAtScreen,
                isFromUser: s.isFromUser, isFinal: s.isFinal
            ))
        }

        if !s.startedLookingAtScreen {
            result.append(SpeechSegment(
                id: s.id, text: s.text, isToScreen: false,
                sentenceStartedLookingAtScreen: false,
                isFromUser: s.isFromUser, isFinal: s.isFinal
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
                        isFromUser: s.isFromUser, isFinal: s.isFinal
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
                        isFromUser: s.isFromUser, isFinal: s.isFinal
                    ))
                }
            }
        }
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
