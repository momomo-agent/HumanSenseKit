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
    public var gazeTrail: [CGPoint] { faceManager.gazeTrail }

    // --- Owned managers ---
    public let faceManager: FaceTrackingManager
    public let audioManager: AudioDetectionManager
    public let handManager: HandGestureManager
    public let deviceMotionManager: DeviceMotionManager
    public let sttManager: STTManager

    private var cancellables = Set<AnyCancellable>()
    private var pendingActivity: HumanActivity?
    private var pendingActivityStartTime: Date?
    private let debounceInterval: TimeInterval = 0.1
    private var previousJawOpen: Float = 0
    private var lastHistoryAppend = Date.distantPast

    public init() {
        self.faceManager = FaceTrackingManager()
        self.audioManager = AudioDetectionManager()
        self.handManager = HandGestureManager()
        self.deviceMotionManager = DeviceMotionManager()
        self.sttManager = STTManager()

        // Wire up face → hand (ARFrame sharing)
        faceManager.handManager = handManager

        // Share audio pipeline: STTManager owns the AudioEngine,
        // AudioDetectionManager receives buffers from it.
        sttManager.audioDetectionManager = audioManager

        setupBindings()
    }

    public func start() {
        faceManager.start()
        deviceMotionManager.start()
        sttManager.start()
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

        // Device → humanState.device
        deviceMotionManager.$deviceState
            .sink { [weak self] device in
                self?.humanState.device = device
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
        let mouthMoving = jawDelta > 0.02 || face.jawOpen > 0.2

        if mouthMoving && audio.isSpeaking {
            let headForward = face.headOrientation.isFacingForward
            let toScreen = face.isLookingAtScreen && headForward
            return toScreen ? .speakingToScreen : .speakingToOther
        }

        if !face.isLookingAtScreen { return .distracted }
        return .listening
    }
}
#endif
