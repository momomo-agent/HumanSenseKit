#if os(iOS)
import Foundation
import Speech
import AVFoundation

@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    /// AudioDetectionManager receives buffers from our shared tap.
    public weak var audioDetectionManager: AudioDetectionManager?

    // --- External inputs (set by Engine) ---
    public var isLookingAtScreen: Bool = false
    public var isSpeaking: Bool = false {
        didSet {
            if isSpeaking {
                gazeAtSpeechOnset = isLookingAtScreen
            }
        }
    }

    private var gazeAtSpeechOnset: Bool = false

    // --- Sentence model ---

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

    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    private var lastCharCount: Int = 0

    // --- Internal state ---
    private var speechStartCaptured: Bool = false
    private var taskGeneration: Int = 0

    // --- Silence detection ---
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private let sentenceGapThreshold: TimeInterval = 1.5

    /// Apple Speech ~60s per-task limit.
    private var taskDurationTimer: Timer?
    private let maxTaskDuration: TimeInterval = 50.0

    public func captureSpeechStartState() {}

    public func start() {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else { return }
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            Task { @MainActor in
                self?.beginListening()
            }
        }
    }

    public func stop() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
        taskGeneration += 1

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil

        finalizeActiveSentence()
        isListening = false
    }

    public func clearSegments() {
        sentences = []
        activeSentence = nil
        rebuildSegments()
    }

    // MARK: - Segment Output

    private func rebuildSegments() {
        var result: [SpeechSegment] = []

        func appendSentence(_ s: Sentence) {
            if !result.isEmpty {
                result.append(SpeechSegment(
                    text: " ",
                    isToScreen: false,
                    sentenceStartedLookingAtScreen: s.startedLookingAtScreen,
                    isFromUser: s.isFromUser
                ))
            }

            if !s.startedLookingAtScreen {
                result.append(SpeechSegment(
                    id: s.id,
                    text: s.text,
                    isToScreen: false,
                    sentenceStartedLookingAtScreen: false,
                    isFromUser: s.isFromUser
                ))
            } else {
                var offset = s.text.startIndex
                for (i, span) in s.gazeSpans.enumerated() {
                    let end = s.text.index(offset, offsetBy: span.charCount, limitedBy: s.text.endIndex) ?? s.text.endIndex
                    let spanText = String(s.text[offset..<end])
                    if !spanText.isEmpty {
                        result.append(SpeechSegment(
                            id: i == 0 ? s.id : span.id,
                            text: spanText,
                            isToScreen: span.isToScreen,
                            sentenceStartedLookingAtScreen: true,
                            isFromUser: s.isFromUser
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
                            isFromUser: s.isFromUser
                        ))
                    }
                }
            }
        }

        for s in sentences { appendSentence(s) }
        if let active = activeSentence, !active.text.isEmpty { appendSentence(active) }

        segments = result
    }

    // MARK: - Sentence Lifecycle

    private func finalizeActiveSentence() {
        guard let active = activeSentence, !active.text.isEmpty else {
            activeSentence = nil
            return
        }
        sentences.append(active)
        if sentences.count > maxSentences {
            sentences.removeFirst()
        }
        activeSentence = nil
        rebuildSegments()
    }

    // MARK: - Audio Engine

    private func beginListening() {
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.recognitionRequest?.append(buffer)
            Task { @MainActor in
                self.audioDetectionManager?.processBuffer(buffer)
            }
        }

        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true

        startRecognitionTask()
        startSilenceTimer()
    }

    // MARK: - Recognition Task

    private func makeRequest() -> SFSpeechAudioBufferRecognitionRequest {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        return request
    }

    private func bindTask(to request: SFSpeechAudioBufferRecognitionRequest, generation: Int) {
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            Task { @MainActor in
                guard generation == self.taskGeneration else {
                    return
                }

                if let result = result {
                    self.handleResult(result)
                    if result.isFinal {
                        self.finalizeActiveSentence()
                        self.startRecognitionTask()
                    }
                }

                if error != nil && !(result?.isFinal ?? false) {
                    self.taskGeneration += 1
                    let gen = self.taskGeneration
                    self.finalizeActiveSentence()
                    if gen == self.taskGeneration {
                        self.startRecognitionTask()
                    }
                }
            }
        }
    }

    private func startRecognitionTask() {
        taskGeneration += 1
        let gen = taskGeneration

        resetActiveSentence()
        resetTaskDurationTimer()

        let request = makeRequest()
        recognitionRequest = request
        bindTask(to: request, generation: gen)
    }

    private func splitSentence() {
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }

        let oldRequest = recognitionRequest
        let oldTask = recognitionTask

        finalizeActiveSentence()

        taskGeneration += 1
        let gen = taskGeneration

        let newRequest = makeRequest()
        recognitionRequest = newRequest

        oldRequest?.endAudio()
        oldTask?.cancel()

        resetActiveSentence()
        resetTaskDurationTimer()
        bindTask(to: newRequest, generation: gen)
    }

    private func resetActiveSentence() {
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [], isFromUser: false)
        lastCharCount = 0
        speechStartCaptured = false
        lastRecognitionTime = nil
    }

    private func resetTaskDurationTimer() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.splitSentence()
            }
        }
    }

    // MARK: - Recognition Result Handling

    private func handleResult(_ result: SFSpeechRecognitionResult) {
        let newText = result.bestTranscription.formattedString
        let newCharCount = newText.count
        let addedChars = max(0, newCharCount - lastCharCount)

        activeSentence?.text = newText
        lastRecognitionTime = Date()

        if !speechStartCaptured && !newText.isEmpty {
            let looking = isSpeaking ? gazeAtSpeechOnset : false
            activeSentence?.startedLookingAtScreen = looking
            activeSentence?.isFromUser = isSpeaking
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
                ensureGazeSpansCoverText(newText)
            }
            lastCharCount = newCharCount
        }

        rebuildSegments()
    }

    // MARK: - Silence Detection

    private func startSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.checkSilence()
            }
        }
    }

    private func checkSilence() {
        guard let lastTime = lastRecognitionTime else { return }
        guard Date().timeIntervalSince(lastTime) >= sentenceGapThreshold else { return }
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }
        splitSentence()
    }

    // MARK: - Gaze Helpers

    private func updateGazeSpans(addedChars: Int) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }

        let lastSpan = spans[spans.count - 1]
        if lastSpan.isToScreen == isLookingAtScreen {
            spans[spans.count - 1] = GazeSpan(
                charCount: lastSpan.charCount + addedChars,
                isToScreen: lastSpan.isToScreen
            )
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
            if spans[lastIdx].charCount == 0 && spans.count > 1 {
                spans.removeLast()
            }
            activeSentence?.gazeSpans = spans
        }
    }
}
#endif
