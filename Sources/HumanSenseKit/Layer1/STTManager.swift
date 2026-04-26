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
    /// Current speaker label from diarization (nil if unavailable).
    @Published public var currentSpeakerLabel: String?

    /// Which speech recognition engine to use.
    public enum BackendType: Sendable {
        /// iOS 26 SpeechAnalyzer (default).
        case speechAnalyzer
        /// Legacy SFSpeechRecognizer (fallback).
        case sfSpeechRecognizer
    }

    /// When true, audio buffers are discarded and STT is effectively paused.
    /// Use this to suppress recognition when the user is not facing the screen.
    public var isMuted: Bool = false

    @Published public var speechDetected: Bool = false

    /// Contextual strings to improve speech recognition accuracy.
    /// Set before calling start(). Examples: app names, person names, technical terms.
    public var contextualStrings: [String] = [] {
        didSet {
            if let backend = speech as? SpeechAnalyzerBackend {
                backend.contextualStrings = contextualStrings
            }
        }
    }

    /// Wall-clock time when the audio stream started (when start() was called).
    /// Use this to convert audio-relative CMTime offsets to system time:
    /// systemTime = audioStreamStartTime + audioOffset
    public private(set) var audioStreamStartTime: Date?

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
        didSet { builder.isSpeaking = isSpeaking }
    }
    /// Onset-weighted gaze score for the current utterance, fed by
    /// HumanStateEngine every frame. SentenceBuilder reads this directly
    /// so the per-sentence score reflects audio-frame-rate sampling, not
    /// the few times STT happens to emit volatile text.
    @Published public var onsetGazeScore: Float = 0 {
        didSet { builder.onsetGazeScore = onsetGazeScore }
    }
    @Published public var onsetFrameCount: Int = 0
    @Published public var onsetLookAtCount: Int = 0
    @Published public var onsetCorrCount: Int = 0

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
                self?.audioStreamStartTime = Date()
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
        audio.onBuffer = { [weak self] buffer in
            guard self?.isMuted != true else { return }
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
        speech.onResult = { [weak self] text, isFinal, speakerLabel, audioStartTime, audioEndTime in
            guard let self else { return }
            self.currentSpeakerLabel = speakerLabel
            self.builder.handleResult(text: text, isFinal: isFinal, audioStartTime: audioStartTime, audioEndTime: audioEndTime)
            if isFinal {
                self.onSentenceFinalized?(text)
                self.builder.resetActive()
            }
            self.rebuildSegments()
        }

        // SpeechDetector VAD callback (only available with SpeechAnalyzer backend)
        if let analyzerBackend = speech as? SpeechAnalyzerBackend {
            analyzerBackend.onSpeechDetected = { [weak self] detected in
                self?.speechDetected = detected
            }
            // Propagate contextual strings set before wireUp
            if !contextualStrings.isEmpty {
                analyzerBackend.contextualStrings = contextualStrings
            }
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
