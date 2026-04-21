#if os(iOS)
import Foundation

/// Layer 2: Sentence building with confirmation semantics.
/// Manages the sentence model, silence-based splitting, gaze span tracking,
/// and the critical distinction between "display text" and "confirmed text".
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
    
    // Silence detection
    private(set) var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private let silenceThreshold: TimeInterval = 2.0
    
    // External inputs
    var isLookingAtScreen: Bool = false
    var isSpeaking: Bool = false {
        didSet {
            if isSpeaking { gazeAtSpeechOnset = isLookingAtScreen }
        }
    }
    private var gazeAtSpeechOnset: Bool = false
    
    /// Called when silence split triggers — caller should split the recognition task.
    var onSilenceSplit: (() -> Void)?
    
    // MARK: - Public API
    
    /// Process a recognition result. Call this from SpeechRecognitionManager.onResult.
    func handleResult(text: String, isFinal: Bool) {
        let newCharCount = text.count
        let addedChars = max(0, newCharCount - lastCharCount)
        
        print("[SentenceBuilder] handleResult: '\(text.prefix(40))' isFinal=\(isFinal) chars=\(newCharCount) added=\(addedChars)")
        
        activeSentence?.text = text
        lastRecognitionTime = Date()
        
        if !speechStartCaptured && !text.isEmpty {
            // When speech starts, isSpeaking may not have updated yet
            // (audio detection lags behind recognition). Default to true
            // if we're receiving text — someone is speaking.
            let speaking = isSpeaking || !text.isEmpty
            let looking = speaking ? (isSpeaking ? gazeAtSpeechOnset : isLookingAtScreen) : false
            activeSentence?.startedLookingAtScreen = looking
            activeSentence?.isFromUser = speaking
            if looking {
                activeSentence?.gazeSpans = [GazeSpan(charCount: newCharCount, isToScreen: isLookingAtScreen)]
            }
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else if addedChars > 0, activeSentence?.startedLookingAtScreen == true {
            updateGazeSpans(addedChars: addedChars)
            lastCharCount = newCharCount
        } else {
            if newCharCount != lastCharCount, activeSentence?.startedLookingAtScreen == true {
                ensureGazeSpansCoverText(text)
            }
            lastCharCount = newCharCount
        }
        
        if isFinal {
            finalizeActiveSentence()
        }
    }
    
    /// Finalize the active sentence and prepare for a new one.
    func finalizeAndReset() {
        finalizeActiveSentence()
        resetActive()
    }
    
    /// Reset active sentence state for a new recognition task.
    func resetActive() {
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [], isFromUser: false)
        lastCharCount = 0
        speechStartCaptured = false
        lastRecognitionTime = nil
    }
    
    /// Clear all sentences and segments.
    func clearAll() {
        sentences = []
        activeSentence = nil
    }
    
    /// Build the current segments array from sentences + active sentence.
    func buildSegments() -> [SpeechSegment] {
        var result: [SpeechSegment] = []
        
        for s in sentences { appendSentence(s, isFinal: true, to: &result) }
        if let active = activeSentence, !active.text.isEmpty {
            appendSentence(active, isFinal: false, to: &result)
        }
        
        return result
    }
    
    // MARK: - Silence Timer
    
    func startSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkSilence() }
        }
    }
    
    func stopSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = nil
    }
    
    private func checkSilence() {
        guard let lastTime = lastRecognitionTime else { return }
        let elapsed = Date().timeIntervalSince(lastTime)
        if elapsed >= silenceThreshold {
            print("[SentenceBuilder] Silence split triggered (elapsed=\(String(format: "%.1f", elapsed))s)")
            finalizeActiveSentence()
            onSilenceSplit?()
        }
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
    
    private func appendSentence(_ s: Sentence, isFinal: Bool, to result: inout [SpeechSegment]) {
        if !result.isEmpty {
            result.append(SpeechSegment(
                text: " ", isToScreen: false,
                sentenceStartedLookingAtScreen: s.startedLookingAtScreen,
                isFromUser: s.isFromUser, isFinal: isFinal
            ))
        }
        
        if !s.startedLookingAtScreen {
            result.append(SpeechSegment(
                id: s.id, text: s.text, isToScreen: false,
                sentenceStartedLookingAtScreen: false,
                isFromUser: s.isFromUser, isFinal: isFinal
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
                        isFromUser: s.isFromUser, isFinal: isFinal
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
                        isFromUser: s.isFromUser, isFinal: isFinal
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
