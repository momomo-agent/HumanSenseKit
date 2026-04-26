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
    private var lastDiagLog: Date = .distantPast
    private var previousJawOpen: Float = 0
    private var lastHistoryAppend = Date.distantPast

    // Speech onset tracker — decides "is this the user speaking?" AND
    // "is the user looking at the screen?" from the first ~500ms of voice
    // activity, using exponential decay on the audio-frame timeline.
    //
    // Both signals (lipCorrelated and lookAtScreen) are sampled in lockstep
    // so gaze and speech verdicts are symmetric: the first few frames
    // dominate, what happens around the onset defines the utterance.
    private let onsetWindow: TimeInterval = 0.5    // sample for 500ms after VAD-on
    private let onsetDecayMs: Float = 150           // e-fold every 150ms
    private let onsetThreshold: Float = 0.30        // weighted corr needed to latch user-speaking
    private var previousVoiceActive: Bool = false
    private var previousInstantSpeaker: Bool = false
    private var onsetStart: TimeInterval? = nil    // set when voice rose; nil outside window
    private var onsetWeightedCorr: Float = 0
    private var onsetWeightedGaze: Float = 0
    private var onsetWeightTotal: Float = 0
    /// Latched verdict for the current speech utterance. Reset on VAD rising edge.
    private var onsetIsUserSpeaking: Bool = false
    /// Weighted gaze score [0,1] for the current utterance (exposed to STT).
    private(set) var onsetGazeScore: Float = 0
    /// Debug counters for the onset window (exposed to UI).
    public private(set) var onsetFrameCount: Int = 0
    public private(set) var onsetLookAtCount: Int = 0
    public private(set) var onsetCorrCount: Int = 0

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
                self?.faceManager.deviceGravity = device.gravity
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
        sttManager.onsetGazeScore = onsetGazeScore
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
        let isCorrNow = lipAudioCorrelator.isCorrelated
        let mouthMoving = jawDelta > 0.02 || face.jawOpen > 0.15 || lipAudioCorrelator.lipActivity > 0.5
        let headForward = face.headOrientation.isFacingForward
        let lookAt = face.isLookingAtScreen

        // Instant "is this person speaking" signal for the onset window.
        // Unlike isCorrelated (needs 500ms warmup), this fires on the first frame.
        let instantSpeaker = mouthMoving && audio.isSpeaking

        // ---- Speech + gaze onset tracker (see field docs above) ----
        let nowTs = ProcessInfo.processInfo.systemUptime
        // Use instantSpeaker (mouth moving + audio) as onset trigger.
        // voiceActive fires on ambient noise before the user speaks;
        // speechDetected (Apple ML VAD) doesn't reliably trigger updateHumanState.
        // instantSpeaker requires actual lip movement, so it fires only when
        // the user is genuinely speaking — and by then they're already looking.
        if instantSpeaker && !previousInstantSpeaker {
            // Mouth+audio rising edge — start a fresh onset window.
            onsetStart = nowTs
            onsetWeightedCorr = 0
            onsetWeightedGaze = 0
            onsetWeightTotal = 0
            onsetIsUserSpeaking = false
            onsetGazeScore = 0
            onsetFrameCount = 0
            onsetLookAtCount = 0
            onsetCorrCount = 0
        }

        if let start = onsetStart {
            let elapsedMs = Float((nowTs - start) * 1000)
            if elapsedMs <= Float(onsetWindow * 1000) {
                // Sample this frame with exponential decay — first frames dominate.
                let w = expf(-elapsedMs / onsetDecayMs)
                onsetWeightedCorr += w * (instantSpeaker ? 1 : 0)
                onsetWeightedGaze += w * (lookAt ? 1 : 0)
                onsetWeightTotal += w
                onsetFrameCount += 1
                if lookAt { onsetLookAtCount += 1 }
                if instantSpeaker { onsetCorrCount += 1 }
                // Latch speaker verdict as soon as the score crosses threshold.
                if !onsetIsUserSpeaking, onsetWeightTotal > 0,
                   onsetWeightedCorr / onsetWeightTotal >= onsetThreshold {
                    onsetIsUserSpeaking = true
                }
                // Publish current gaze score continuously (stabilizes as the
                // window fills). STT will read whatever is latest when it
                // emits text.
                if onsetWeightTotal > 0 {
                    onsetGazeScore = onsetWeightedGaze / onsetWeightTotal
                    sttManager.onsetGazeScore = onsetGazeScore
                    sttManager.onsetFrameCount = onsetFrameCount
                    sttManager.onsetLookAtCount = onsetLookAtCount
                    sttManager.onsetCorrCount = onsetCorrCount
                }
            } else if !voiceActive {
                // Utterance ended — keep verdict so late-arriving STT can use it,
                // but stop sampling. Reset happens on the next rising edge.
                onsetStart = nil
            }
        }

        if !voiceActive {
            // Gradual reset when voice fades (next rising edge re-arms everything).
            onsetStart = nil
        }

        previousVoiceActive = voiceActive
        previousInstantSpeaker = instantSpeaker

        // ---- Build the user-speaking gate ----
        // Prefer the onset verdict (captured from the first ~500ms); fall back
        // to the instantaneous correlator for long utterances where the onset
        // window has already closed.
        let isUserSpeaking = voiceActive && (onsetIsUserSpeaking || isCorrNow)

        // Rate-limited diagnostic log: full gate state every ~0.5s.
        let now = Date()
        if now.timeIntervalSince(lastDiagLog) > 0.5 {
            let corrScore = onsetWeightTotal > 0 ? onsetWeightedCorr / onsetWeightTotal : 0
            print(String(format:
                "[HSE] voice=%@ corr=%@(%.2f) onsetSpk=%@(%.2f) onsetGaze=%.2f lookAt=%@ headFwd=%@",
                voiceActive ? "✓" : "·",
                isCorrNow ? "✓" : "·",
                lipAudioCorrelator.correlation,
                onsetIsUserSpeaking ? "✓" : "·",
                corrScore,
                onsetGazeScore,
                lookAt ? "✓" : "·",
                headForward ? "✓" : "·"))
            lastDiagLog = now
        }

        if isUserSpeaking {
            let toScreen = lookAt && headForward
            return toScreen ? .speakingToScreen : .speakingToOther
        }

        if !lookAt { return .distracted }
        return .listening
    }
}
#endif
