#if os(iOS)
import Foundation
import ARKit
import Combine

@MainActor
public class FaceTrackingManager: NSObject, ObservableObject {
    @Published public var faceState = FaceState()
    @Published public var currentAnchor: ARFaceAnchor?
    @Published public var currentFrame: ARFrame?
    @Published public var gazeTrail: [CGPoint] = []

    /// The ARSession this manager reads from. Owned externally; FaceTrackingManager is a consumer.
    private var arSession: ARSession?
    private let processingQueue = DispatchQueue(label: "com.momomo.facetracking", qos: .userInitiated)

    private var gazeFilterX: LowPassFilter?
    private var gazeFilterY: LowPassFilter?
    private var previousJawOpen: Float = 0
    private var lastTrailAppend = Date.distantPast
    private let headGestureDetector = HeadGestureDetector()
    private let emotionDetector = EmotionDetector()
    nonisolated(unsafe) private var noFaceFrames: Int = 0
    nonisolated(unsafe) private var untrackedFrames: Int = 0
    private let noFaceThreshold = 5

    public weak var handManager: HandGestureManager?

    public override init() {
        super.init()
    }

    /// Attach to an externally-owned ARSession.
    /// FaceTrackingManager becomes its delegate and configures face tracking.
    public func start(session: ARSession) {
        self.arSession = session
        guard ARFaceTrackingConfiguration.isSupported else {
            NSLog("[FaceTracking] ARFaceTracking NOT supported on this device")
            return
        }
        NSLog("[FaceTracking] Starting face tracking on shared ARSession")
        let config = ARFaceTrackingConfiguration()
        config.worldAlignment = .camera
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    /// Legacy convenience — creates its own ARSession if none provided.
    public func start() {
        let session = ARSession()
        start(session: session)
    }

    public func stop() {
        arSession?.pause()
    }

    nonisolated private func extractHeadOrientation(from transform: simd_float4x4) -> (yaw: Float, pitch: Float, roll: Float) {
        let yaw = atan2(transform.columns.0.z, transform.columns.2.z)
        let pitch = asin(-transform.columns.1.z)
        let rawRoll = atan2(transform.columns.1.x, transform.columns.1.y)
        let roll = rawRoll + 1.6
        return (yaw, pitch, roll)
    }
}

extension FaceTrackingManager: ARSessionDelegate {
    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let anchor = frame.anchors.first as? ARFaceAnchor else {
            noFaceFrames += 1
            if noFaceFrames >= noFaceThreshold {
                Task { @MainActor in
                    self.faceState.faceDetected = false
                    self.faceState.isLookingAtScreen = false
                }
            }
            return
        }

        noFaceFrames = 0

        if !anchor.isTracked {
            untrackedFrames += 1
            if untrackedFrames >= noFaceThreshold {
                Task { @MainActor in
                    self.faceState.faceDetected = false
                    self.faceState.isLookingAtScreen = false
                }
            }
            return
        }
        untrackedFrames = 0

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            let (yaw, pitch, roll) = self.extractHeadOrientation(from: anchor.transform)

            let lookAtVector = anchor.transform * SIMD4<Float>(anchor.lookAtPoint, 1)

            guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let orientation = scene.windows.first?.windowScene?.interfaceOrientation else { return }

            let lookPoint = frame.camera.projectPoint(
                SIMD3<Float>(x: lookAtVector.x, y: lookAtVector.y, z: lookAtVector.z),
                orientation: orientation,
                viewportSize: UIScreen.main.bounds.size
            )

            let size = UIScreen.main.bounds.size
            let adjusted = (x: size.width - lookPoint.x, y: size.height - lookPoint.y)

            let bs = anchor.blendShapes
            let jawOpen = bs[.jawOpen]?.floatValue ?? 0
            let mouthClose = bs[.mouthClose]?.floatValue ?? 0
            let eyeBlinkL = bs[.eyeBlinkLeft]?.floatValue ?? 0
            let eyeBlinkR = bs[.eyeBlinkRight]?.floatValue ?? 0

            let eyeLookInL = bs[.eyeLookInLeft]?.floatValue ?? 0
            let eyeLookOutL = bs[.eyeLookOutLeft]?.floatValue ?? 0
            let eyeLookUpL = bs[.eyeLookUpLeft]?.floatValue ?? 0
            let eyeLookDownL = bs[.eyeLookDownLeft]?.floatValue ?? 0

            let eyeLookInR = bs[.eyeLookInRight]?.floatValue ?? 0
            let eyeLookOutR = bs[.eyeLookOutRight]?.floatValue ?? 0
            let eyeLookUpR = bs[.eyeLookUpRight]?.floatValue ?? 0
            let eyeLookDownR = bs[.eyeLookDownRight]?.floatValue ?? 0

            Task { @MainActor in
                if self.gazeFilterX == nil {
                    self.gazeFilterX = LowPassFilter(value: adjusted.x)
                    self.gazeFilterY = LowPassFilter(value: adjusted.y)
                } else {
                    self.gazeFilterX?.update(with: adjusted.x)
                    self.gazeFilterY?.update(with: adjusted.y)
                }

                var newState = FaceState()
                newState.faceDetected = true
                newState.gazePoint = CGPoint(x: self.gazeFilterX?.value ?? adjusted.x,
                                            y: self.gazeFilterY?.value ?? adjusted.y)

                newState.headYaw = yaw
                newState.headPitch = pitch
                newState.headRoll = roll

                newState.headGesture = self.headGestureDetector.update(yaw: yaw, pitch: pitch, roll: roll)

                newState.emotion = self.emotionDetector.detectEmotion(from: anchor.blendShapes, isSpeaking: false)

                newState.distanceFromCamera = abs(anchor.transform.columns.3.z)

                let screenSize = UIScreen.main.bounds.size
                let marginRatio: CGFloat = 0.1
                let marginX = screenSize.width * marginRatio
                let marginY = screenSize.height * marginRatio
                let gazeX = self.gazeFilterX?.value ?? adjusted.x
                let gazeY = self.gazeFilterY?.value ?? adjusted.y
                let gazeInCenter = gazeX > marginX && gazeX < screenSize.width - marginX &&
                                   gazeY > marginY && gazeY < screenSize.height - marginY
                newState.isLookingAtScreen = gazeInCenter

                newState.jawOpen = jawOpen
                newState.mouthClose = mouthClose

                newState.eyeBlinkLeft = eyeBlinkL
                newState.eyeBlinkRight = eyeBlinkR

                newState.eyeLookInLeft = eyeLookInL
                newState.eyeLookOutLeft = eyeLookOutL
                newState.eyeLookUpLeft = eyeLookUpL
                newState.eyeLookDownLeft = eyeLookDownL

                newState.eyeLookInRight = eyeLookInR
                newState.eyeLookOutRight = eyeLookOutR
                newState.eyeLookUpRight = eyeLookUpR
                newState.eyeLookDownRight = eyeLookDownR

                self.faceState = newState
                self.previousJawOpen = jawOpen
                self.currentAnchor = anchor
                self.currentFrame = frame

                // Append gaze trail at ~10fps
                let now = Date()
                if now.timeIntervalSince(self.lastTrailAppend) >= 0.1 {
                    self.gazeTrail.append(newState.gazePoint)
                    if self.gazeTrail.count > 100 { self.gazeTrail.removeFirst() }
                    self.lastTrailAppend = now
                }
            }
        }
    }

    nonisolated public func session(_ session: ARSession, didFailWithError error: Error) {
        NSLog("[FaceTracking] ARSession failed: %@", error.localizedDescription)
    }

    nonisolated public func sessionWasInterrupted(_ session: ARSession) {
        NSLog("[FaceTracking] ARSession interrupted")
    }

    nonisolated public func sessionInterruptionEnded(_ session: ARSession) {
        NSLog("[FaceTracking] ARSession interruption ended, restarting")
        Task { @MainActor in
            self.start()
        }
    }
}
#endif
