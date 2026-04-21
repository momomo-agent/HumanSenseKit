#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// STTManager — thin orchestrator composing three layers:
///   Layer 0: AudioEngineManager  (audio lifecycle)
///   Layer 1: SpeechRecognitionManager  (SFSpeechRecognizer)
///   Layer 2: SentenceBuilder  (sentence model + gaze tracking)
@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    @Published public var lastError: String?

    private let audio = AudioEngineManager()
    private let speech = SpeechRecognitionManager()
    private let builder = SentenceBuilder()

    public var audioEngine: AVAudioEngine {
        get { audio.audioEngine }
        set { audio.audioEngine = newValue }
    }

    public var usesExternalEngine: Bool {
        get { audio.usesExternalEngine }
        set { audio.usesExternalEngine = newValue }
    }

    public weak var audioDetectionManager: AudioDetectionManager?

    public var isLookingAtScreen: Bool = false {
        didSet { builder.isLookingAtScreen = isLookingAtScreen }
    }
    public var isSpeaking: Bool = false {
        didSet { builder.isSpeaking = isSpeaking }
    }

    public func captureSpeechStartState() {}

    // MARK: - Lifecycle

    public func start() {
        print("[STT] start()")
        speech.authorize { [weak self] authorized in
            guard authorized else {
                print("[STT] Not authorized")
                return
            }
            Task { @MainActor in
                guard let self else { return }
                self.lastError = nil
                self.wireUp()
                self.audio.start()
                self.speech.startTask()
                self.builder.resetActive()
                self.isListening = true
            }
        }
    }

    public func stop() {
        speech.stop()
        audio.stop()
        builder.finalizeAndReset()
        rebuildSegments()
        isListening = false
    }

    public func clearSegments() {
        builder.clearAll()
        rebuildSegments()
    }

    // MARK: - Wiring

    private func wireUp() {
        // Layer 0 → Layer 1: audio buffers feed recognition
        audio.onBuffer = { [weak self] buffer in
            self?.speech.appendBuffer(buffer)
            Task { @MainActor in
                self?.audioDetectionManager?.processBuffer(buffer)
            }
        }

        // Layer 0: engine restart → restart recognition
        audio.onRestart = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.speech.startTask()
                self.builder.resetActive()
            }
        }

        // Layer 1 → Layer 2: transcription results
        speech.onResult = { [weak self] text, isFinal in
            guard let self else { return }
            self.builder.handleResult(text: text, isFinal: isFinal)
            if isFinal {
                // SFSpeechRecognizer finalized — restart for next utterance
                self.speech.startTask()
                self.builder.resetActive()
            }
            self.rebuildSegments()
        }

        speech.onError = { [weak self] error in
            self?.lastError = error.localizedDescription
            self?.builder.finalizeAndReset()
            self?.rebuildSegments()
        }
    }

    private func rebuildSegments() {
        segments = builder.buildSegments()
    }
}
#endif
