#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// STTManager — thin orchestrator composing three layers:
///   Layer 0: AudioEngineManager  (audio lifecycle)
///   Layer 1: SpeechRecognitionBackend  (SpeechAnalyzer or SFSpeechRecognizer)
///   Layer 2: SentenceBuilder  (sentence model + gaze tracking)
@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    @Published public var lastError: String?

    /// Which speech recognition engine to use.
    public enum BackendType: Sendable {
        /// iOS 26 SpeechAnalyzer (default).
        case speechAnalyzer
        /// Legacy SFSpeechRecognizer (fallback).
        case sfSpeechRecognizer
    }

    /// Set a closure to capture signal snapshots for debug display on each segment.
    public var captureSignals: (() -> SpeechSegment.SignalSnapshot)? {
        get { builder.captureSignals }
        set { builder.captureSignals = newValue }
    }

    // --- Sub-components ---
    private let audio = AudioEngineManager()
    private let speech: any SpeechRecognitionBackend
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
        didSet {
            builder.isSpeaking = isSpeaking
            // Rebuild segments so isFromUser reflects new state immediately
            if oldValue != isSpeaking {
                rebuildSegments()
            }
        }
    }

    public func captureSpeechStartState() {}

    // MARK: - Init

    /// Create an STTManager with the specified backend.
    /// Defaults to `.speechAnalyzer` (iOS 26).
    public init(backend: BackendType = .speechAnalyzer) {
        switch backend {
        case .speechAnalyzer:
            self.speech = SpeechAnalyzerBackend()
        case .sfSpeechRecognizer:
            self.speech = SFSpeechBackend()
        }
        super.init()
    }

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
                self?.isListening = true
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
        // Capture sub-components directly — avoid accessing @MainActor self from audio thread
        let speech = self.speech
        weak var audioDetection = self.audioDetectionManager

        // Layer 0 → Layer 1: audio buffers feed recognition
        audio.onBuffer = { buffer in
            speech.appendBuffer(buffer)
            Task { @MainActor in
                audioDetection?.processBuffer(buffer)
            }
        }

        // Layer 0: engine restart → restart recognition
        audio.onRestart = { [weak self] in
            Task { @MainActor in
                self?.speech.startTask()
                self?.builder.resetActive()
            }
        }

        // Layer 1 → Layer 2: transcription results feed sentence builder
        speech.onResult = { [weak self] text, isFinal in
            guard let self else { return }
            self.builder.handleResult(text: text, isFinal: isFinal)
            if isFinal {
                // Apple finalized this segment — reset for next sentence
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
