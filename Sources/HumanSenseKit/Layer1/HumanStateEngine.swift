import Foundation
import Combine

@MainActor
@Observable
class HumanStateEngine {
    var humanState = HumanState()
    var stateHistory: [(date: Date, activity: HumanActivity)] = []

    private let faceManager: FaceTrackingManager
    private let audioManager: AudioDetectionManager
    private let handManager: HandGestureManager

    private var cancellables = Set<AnyCancellable>()
    private var pendingActivity: HumanActivity?
    private var pendingActivityStartTime: Date?
    private let debounceInterval: TimeInterval = 0.1
    private var previousJawOpen: Float = 0
    private var lastHistoryAppend = Date.distantPast

    init(faceManager: FaceTrackingManager, audioManager: AudioDetectionManager, handManager: HandGestureManager) {
        self.faceManager = faceManager
        self.audioManager = audioManager
        self.handManager = handManager

        faceManager.$faceState
            .combineLatest(audioManager.$audioState)
            .sink { [weak self] face, audio in
                self?.updateHumanState(face: face, audio: audio)
            }
            .store(in: &cancellables)

        handManager.$handState
            .sink { [weak self] hand in
                self?.humanState.hand = hand
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
        humanState.debugJawDelta = jawDelta  // Expose for debugging
        
        let mouthMoving = jawDelta > 0.02 || face.jawOpen > 0.2  // Back to OR
        
        // Speaking requires BOTH mouth movement AND audio
        // Only check audio if mouth is moving
        if mouthMoving {
            if audio.isSpeaking { return .speaking }
        }

        if !face.isLookingAtScreen { return .distracted }
        return .listening
    }

    func start() {
        faceManager.start()
        audioManager.start()
        handManager.start()
    }

    func stop() {
        faceManager.stop()
        audioManager.stop()
        handManager.stop()
    }
}
