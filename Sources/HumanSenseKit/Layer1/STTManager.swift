#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// STTManager — thin orchestrator composing three layers:
///   Layer 0: AudioEngineManager  (audio lifecycle)
///   Layer 1: SpeechRecognitionManager  (recognition tasks)
///   Layer 2: SentenceBuilder  (sentence model + confirmation)
///
/// Public API is unchanged — segments, isListening, start/stop, etc.
@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    @Published public var lastError: String?

    // --- Sub-components ---
    private let audio = AudioEngineManager()
    private let speech = SpeechRecognitionManager()
    private let builder = SentenceBuilder()

    /// The audio engine. Can be replaced before calling start().
    public var audioEngine: AVAudioEngine {
        get { audio.audioEngine }
        set { audio.audioEngine = newValue }
    }

    /// When true, STTManager will NOT manage the audio engine lifecycle.
    public var usesExternalEngine: Bool {
        get { audio.usesExternalEngine }
        set { audio.usesExternalEngine = newValue }
    }

    /// AudioDetectionManager receives buffers from our shared tap.
    public weak var audioDetectionManager: AudioDetectionManager?

    // --- External inputs (set by Engine) ---
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
                self?.lastError = nil
                self?.wireUp()
                self?.audio.start()
                self?.speech.startTask()
                self?.builder.resetActive()
                self?.builder.startSilenceTimer()
                self?.isListening = true
            }
        }
    }

    public func stop() {
        builder.stopSilenceTimer()
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
                self?.speech.startTask()
                self?.builder.resetActive()
            }
        }

        // Layer 1 → Layer 2: recognition results feed sentence builder
        speech.onResult = { [weak self] result in
            guard let self else { return }
            let text = result.bestTranscription.formattedString
            self.builder.handleResult(text: text, isFinal: result.isFinal)
            if result.isFinal {
                self.builder.resetActive()
            }
            self.rebuildSegments()
        }

        speech.onError = { [weak self] error in
            self?.lastError = error.localizedDescription
            self?.builder.finalizeAndReset()
            self?.rebuildSegments()
        }

        // Layer 1: task duration limit → split
        speech.onTaskSplit = { [weak self] in
            self?.performSplit()
        }

        // Layer 2 → Layer 1: silence split → new recognition task
        builder.onSilenceSplit = { [weak self] in
            self?.performSplit()
        }
    }

    private func performSplit() {
        print("[STT] performSplit — finalizing and splitting task")
        builder.finalizeAndReset()
        speech.splitTask()
        builder.resetActive()
        rebuildSegments()
    }

    private func rebuildSegments() {
        segments = builder.buildSegments()
    }
}
#endif
