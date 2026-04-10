#if os(iOS)
import Foundation
import Speech
import AVFoundation

@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    @Published public var lastError: String?

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
        NSLog("[STT] start() called")
        guard let recognizer = speechRecognizer else {
            NSLog("[STT] Speech recognizer is nil!")
            return
        }
        NSLog("[STT] recognizer.isAvailable=%d, locale=%@", recognizer.isAvailable ? 1 : 0, recognizer.locale.identifier)
        guard recognizer.isAvailable else {
            NSLog("[STT] Speech recognizer not available")
            return
        }
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            NSLog("[STT] Authorization status: %d (0=notDetermined, 1=denied, 2=restricted, 3=authorized)", status.rawValue)
            guard status == .authorized else {
                NSLog("[STT] Not authorized, aborting")
                return
            }
            Task { @MainActor in
                NSLog("[STT] Authorized, calling beginListening()")
                self?.lastError = nil
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
        // Observe audio session interruptions (ARKit can steal the session)
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.interruptionNotification, object: nil)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
        
        configureAndStartAudioEngine()
    }
    
    @objc private func handleAudioInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }
        
        if type == .ended {
            NSLog("[STT] Audio interruption ended — restarting audio engine")
            Task { @MainActor in
                self.configureAndStartAudioEngine()
            }
        } else {
            NSLog("[STT] Audio interruption began")
        }
    }
    
    private func configureAndStartAudioEngine() {
        // Stop existing engine and clean up
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        // Always remove tap before reinstalling
        audioEngine.inputNode.removeTap(onBus: 0)
        
        let audioSession = AVAudioSession.sharedInstance()
        // Use the existing audio session category if ARKit already configured it.
        // Only set category if not already playAndRecord (ARKit sets this for face tracking).
        do {
            let currentCategory = audioSession.category
            if currentCategory != .playAndRecord {
                try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers])
                NSLog("[STT] Audio session configured: playAndRecord (was %@)", currentCategory.rawValue)
            } else {
                NSLog("[STT] Audio session already playAndRecord, reusing")
            }
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            NSLog("[STT] Audio session error: %@", error.localizedDescription)
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            NSLog("[STT] Invalid recording format: sampleRate=%.0f channels=%d", recordingFormat.sampleRate, recordingFormat.channelCount)
            return
        }

        NSLog("[STT] Recording format: sampleRate=%.0f channels=%d", recordingFormat.sampleRate, recordingFormat.channelCount)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.recognitionRequest?.append(buffer)
            Task { @MainActor in
                self.audioDetectionManager?.processBuffer(buffer)
            }
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            NSLog("[STT] Audio engine started successfully")
        } catch {
            NSLog("[STT] Audio engine start failed: %@", error.localizedDescription)
        }
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
                    NSLog("[STT] Stale callback gen=%d current=%d, ignoring", generation, self.taskGeneration)
                    return
                }

                if let result = result {
                    NSLog("[STT] Got result: '%@' isFinal=%d", result.bestTranscription.formattedString, result.isFinal ? 1 : 0)
                    self.handleResult(result)
                    if result.isFinal {
                        self.finalizeActiveSentence()
                        self.startRecognitionTask()
                    }
                }

                if let error = error, !(result?.isFinal ?? false) {
                    NSLog("[STT] Recognition error: %@ (code=%d)", error.localizedDescription, (error as NSError).code)
                    self.lastError = error.localizedDescription
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
        
        NSLog("[STT] startRecognitionTask gen=%d, recognizer=%@, onDevice=%d", gen,
              speechRecognizer != nil ? "exists" : "nil",
              request.requiresOnDeviceRecognition ? 1 : 0)
        
        bindTask(to: request, generation: gen)
        
        NSLog("[STT] recognitionTask=%@", recognitionTask != nil ? "created" : "nil")
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
