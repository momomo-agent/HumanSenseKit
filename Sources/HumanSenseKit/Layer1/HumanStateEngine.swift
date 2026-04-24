#if os(iOS)
import Foundation
import ARKit
import Combine

@MainActor
@Observable
public class HumanStateEngine {
    public var humanState = HumanState()
    public var stateHistory: [(date: Date, activity: HumanActivity)] = []

    // Expose for views that need ARFaceAnchor (FaceMeshView)
    public var currentFaceAnchor: ARFaceAnchor? { faceManager.currentAnchor }
    /// The latest ARFrame — use for AvatarKit rendering and camera preview.
    public var currentARFrame: ARFrame? { faceManager.currentFrame }
    public var gazeTrail: [CGPoint] { faceManager.gazeTrail }

    /// Real-time ARFrame callback — bypasses SwiftUI for 60fps consumers.
    public var onARFrame: ((ARFrame) -> Void)? {
        get { faceManager.onARFrame }
        set { faceManager.onARFrame = newValue }
    }

    // --- Owned managers ---
    public let faceManager: FaceTrackingManager
    public let audioManager: AudioDetectionManager
    public let handManager: HandGestureManager
    public let deviceMotionManager: DeviceMotionManager
    public let sttManager: STTManager
    public let lipAudioCorrelator = LipAudioCorrelator()

    private var cancellables = Set<AnyCancellable>()
    private var pendingActivity: HumanActivity?
    private var pendingActivityStartTime: Date?
    private let debounceInterval: TimeInterval = 0.1
    private var previousJawOpen: Float = 0
    private var lastHistoryAppend = Date.distantPast

    public init(sttBackend: STTManager.BackendType = .speechAnalyzer) {
        self.faceManager = FaceTrackingManager()
        self.audioManager = AudioDetectionManager()
        self.handManager = HandGestureManager()
        self.deviceMotionManager = DeviceMotionManager()
        self.sttManager = STTManager(backend: sttBackend)

        // Wire up face → hand (ARFrame sharing)
        faceManager.handManager = handManager

        // Share audio pipeline: STTManager owns the AudioEngine,
        // AudioDetectionManager receives buffers from it.
        sttManager.audioDetectionManager = audioManager

        // Wire signal snapshot capture for debug display
        sttManager.captureSignals = { [weak self] in
            guard let self else { return SpeechSegment.SignalSnapshot() }
            let face = self.humanState.face
            let audio = self.humanState.audio
            let jawDelta = abs(face.jawOpen - self.previousJawOpen)
            return SpeechSegment.SignalSnapshot(
                mouthMoving: jawDelta > 0.02 || face.jawOpen > 0.15 || self.lipAudioCorrelator.lipActivity > 0.5,
                audioActive: audio.isSpeaking,
                gazeOnScreen: face.isLookingAtScreen,
                headForward: face.headOrientation.isFacingForward,
                lipCorrelation: self.lipAudioCorrelator.correlation,
                lipCorrelated: self.lipAudioCorrelator.isCorrelated,
                activity: self.humanState.activity.rawValue,
                bestOffset: self.lipAudioCorrelator.bestOffset,
                waveform: self.lipAudioCorrelator.samplePoints
            )
        }

        // Dump correlator window on sentence finalization for debug
        sttManager.onSentenceFinalized = { [weak self] text in
            guard let self else { return }
            let dump = self.lipAudioCorrelator.dumpWindow()
            print("[LipCorr] Sentence: '\(text.prefix(40))' r=\(String(format: "%.3f", self.lipAudioCorrelator.correlation))")
            print(dump)
        }

        setupBindings()
    }

    public func start(session: ARSession? = nil) {
        NSLog("[Engine] start() called, session=%@", session != nil ? "provided" : "nil")
        // Start ARKit first — it reconfigures AVAudioSession
        if let session {
            faceManager.start(session: session)
        } else {
            faceManager.start()
        }
        deviceMotionManager.start()
        
        NSLog("[Engine] Subscribing to arSessionReady (current=%d)", faceManager.arSessionReady ? 1 : 0)
        // Start STT only after ARKit is fully running (first frame received).
        // This ensures ARKit's audio session reconfiguration is done before
        // STT configures its own audio engine on top.
        faceManager.$arSessionReady
            .first(where: { $0 })
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                NSLog("[Engine] ARSession ready — starting STT")
                self?.sttManager.start()
            }
            .store(in: &cancellables)
    }

    public func stop() {
        faceManager.stop()
        audioManager.stop()
        deviceMotionManager.stop()
        sttManager.stop()
    }

    // MARK: - Private

    private func setupBindings() {
        // Face + Audio → activity inference + STT sync
        faceManager.$faceState
            .combineLatest(audioManager.$audioState)
            .sink { [weak self] face, audio in
                self?.updateHumanState(face: face, audio: audio)
            }
            .store(in: &cancellables)

        // Hand → humanState.hand
        handManager.$handState
            .sink { [weak self] hand in
                self?.humanState.hand = hand
            }
            .store(in: &cancellables)

        // Device → humanState.device + faceTracking gravity
        deviceMotionManager.$deviceState
            .sink { [weak self] device in
                self?.humanState.device = device
                self?.faceTrackingManager.deviceGravity = device.gravity
            }
            .store(in: &cancellables)

        // STT → humanState.speech (for non-UI consumers)
        sttManager.$segments
            .combineLatest(sttManager.$isListening)
            .sink { [weak self] segments, isListening in
                self?.humanState.speech = SpeechState(segments: segments, isListening: isListening)
            }
            .store(in: &cancellables)
    }

    private func updateHumanState(face: FaceState, audio: AudioState) {
        humanState.face = face
        humanState.audio = audio

        let newActivity = inferActivity(face: face, audio: audio)

        if newActivity != humanState.activity {
            if pendingActivity == newActivity {
                if let startTime = pendingActivityStartTime,
                   Date().timeIntervalSince(startTime) >= debounceInterval {
                    humanState.activity = newActivity
                    pendingActivity = nil
                    pendingActivityStartTime = nil
                }
            } else {
                pendingActivity = newActivity
                pendingActivityStartTime = Date()
            }
        } else {
            pendingActivity = nil
            pendingActivityStartTime = nil
        }

        // Sync state to STT manager
        sttManager.isLookingAtScreen = face.isLookingAtScreen
        let activitySpeaking = humanState.activity.isSpeaking
        if sttManager.isSpeaking != activitySpeaking {
            sttManager.isSpeaking = activitySpeaking
            if activitySpeaking {
                sttManager.captureSpeechStartState()
            }
        }

        // History
        let now = Date()
        if now.timeIntervalSince(lastHistoryAppend) >= 0.1 {
            stateHistory.append((date: now, activity: humanState.activity))
            let cutoff = now.addingTimeInterval(-10)
            stateHistory.removeAll { $0.date < cutoff }
            lastHistoryAppend = now
        }

        previousJawOpen = face.jawOpen
    }

    private func inferActivity(face: FaceState, audio: AudioState) -> HumanActivity {
        if !face.faceDetected { return .absent }
        if face.eyesClosed { return .eyesClosed }

        let jawDelta = abs(face.jawOpen - previousJawOpen)
        // Feed lip-audio correlator every frame with all lip blendshapes
        let lipFrame = LipAudioCorrelator.LipFrame(
            jawOpen: face.jawOpen,
            mouthClose: face.mouthClose,
            mouthFunnel: face.mouthFunnel,
            mouthPucker: face.mouthPucker,
            mouthLeft: face.mouthLeft,
            mouthRight: face.mouthRight,
            mouthStretchLeft: face.mouthStretchLeft,
            mouthStretchRight: face.mouthStretchRight
        )
        lipAudioCorrelator.addSample(face: lipFrame, audioRMS: audio.volume)

        // Speech detection: use Apple's SpeechDetector (ML-based VAD) when available,
        // fall back to audio.isSpeaking (RMS threshold).
        // Combined with lip-audio correlation for "is it THIS person speaking?"
        let voiceActive = sttManager.speechDetected || audio.isSpeaking
        if voiceActive && lipAudioCorrelator.isCorrelated {
            let headForward = face.headOrientation.isFacingForward
            let toScreen = face.isLookingAtScreen && headForward
            return toScreen ? .speakingToScreen : .speakingToOther
        }

        if !face.isLookingAtScreen { return .distracted }
        return .listening
    }
}
#endif
