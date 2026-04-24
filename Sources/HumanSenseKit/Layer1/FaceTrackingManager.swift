#if os(iOS)
import Foundation
import ARKit
import Combine

@MainActor
public class FaceTrackingManager: NSObject, ObservableObject {
    @Published public var faceState = FaceState()
    /// Latest ARFaceAnchor — updated synchronously on ARKit delegate thread.
    public nonisolated(unsafe) var currentAnchor: ARFaceAnchor?
    /// Latest ARFrame — updated synchronously on ARKit delegate thread.
    public nonisolated(unsafe) var currentFrame: ARFrame?
    @Published public var gazeTrail: [CGPoint] = []
    @Published public var arSessionReady = false
    
    /// Real-time ARFrame callback for consumers (e.g. AvatarKit).
    /// Called synchronously on ARKit's delegate thread at ~60fps.
    public nonisolated(unsafe) var onARFrame: ((ARFrame) -> Void)?

    /// The ARSession this manager reads from. Owned externally; FaceTrackingManager is a consumer.
    private var arSession: ARSession?

    private var gazeFilterX: LowPassFilter?
    private var gazeFilterY: LowPassFilter?
    private var previousJawOpen: Float = 0
    private var lastTrailAppend = Date.distantPast
    private let headGestureDetector = HeadGestureDetector()
    private let emotionDetector = EmotionDetector()
    nonisolated(unsafe) private var noFaceFrames: Int = 0
    nonisolated(unsafe) private var untrackedFrames: Int = 0
    nonisolated(unsafe) private var hasSignaledReady: Bool = false
    nonisolated(unsafe) private var cachedOrientation: UIInterfaceOrientation = .portrait
    nonisolated(unsafe) private var cachedScreenSize: CGSize = CGSize(width: 390, height: 844)
    /// Device gravity vector for gaze posture compensation.
    /// Updated from DeviceMotionManager. gravity.z ≈ -1 means face-up (lying), gravity.y ≈ -1 means upright.
    nonisolated(unsafe) public var deviceGravity: (x: Double, y: Double, z: Double) = (0, -1, 0)
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
        config.providesAudioData = false
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
        // Signal that ARKit's audio session setup is done (first frame = session fully running)
        if !hasSignaledReady {
            hasSignaledReady = true
            Task { @MainActor in
                self.arSessionReady = true
                if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                    self.cachedOrientation = scene.windows.first?.windowScene?.interfaceOrientation ?? .portrait
                }
                self.cachedScreenSize = UIScreen.main.bounds.size
            }
        }

        guard let anchor = frame.anchors.first as? ARFaceAnchor else {
            noFaceFrames += 1
            // Overwrite synchronously — only ever 1 frame retained
            self.currentFrame = frame
            self.onARFrame?(frame)
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

        // --- All extraction done synchronously on ARKit's delegate thread ---
        // No async queue = no closures retaining ARFrame

        let (yaw, pitch, roll) = extractHeadOrientation(from: anchor.transform)
        let lookAtVector = anchor.transform * SIMD4<Float>(anchor.lookAtPoint, 1)
        let orientation = cachedOrientation
        let size = cachedScreenSize

        let lookPoint = frame.camera.projectPoint(
            SIMD3<Float>(x: lookAtVector.x, y: lookAtVector.y, z: lookAtVector.z),
            orientation: orientation,
            viewportSize: size
        )

        let adjustedX = size.width - lookPoint.x
        let adjustedY = size.height - lookPoint.y

        // Face-camera angle compensation:
        // When face is not perpendicular to device (e.g. lying down, phone on desk),
        // ARKit's gaze projection drifts on Y-axis.
        // Use face pitch (from faceAnchor.transform) to compensate.
        // pitch ≈ 0 = face perpendicular to camera (ideal)
        // pitch > 0.35 (~20°) = face tilted relative to camera
        let facePitch = pitch  // already extracted from anchor.transform
        let tiltThreshold: Float = 0.35  // ~20°, below this no compensation
        let compensatedY: CGFloat
        if abs(facePitch) > tiltThreshold {
            // How far from ideal (0°) toward extreme (~60°)
            let maxPitch: Float = 1.05  // ~60°
            let tiltRatio = CGFloat(min((abs(facePitch) - tiltThreshold) / (maxPitch - tiltThreshold), 1.0))
            let centerY = size.height / 2
            // Blend gaze toward center proportional to face-camera angle
            compensatedY = adjustedY + (centerY - adjustedY) * tiltRatio * 0.35
        } else {
            compensatedY = adjustedY
        }
        let compensatedX = adjustedX

        let bs = anchor.blendShapes
        let jawOpen = bs[.jawOpen]?.floatValue ?? 0
        let mouthClose = bs[.mouthClose]?.floatValue ?? 0
        let mouthFunnel = bs[.mouthFunnel]?.floatValue ?? 0
        let mouthPucker = bs[.mouthPucker]?.floatValue ?? 0
        let mouthLeft = bs[.mouthLeft]?.floatValue ?? 0
        let mouthRight = bs[.mouthRight]?.floatValue ?? 0
        let mouthStretchLeft = bs[.mouthStretchLeft]?.floatValue ?? 0
        let mouthStretchRight = bs[.mouthStretchRight]?.floatValue ?? 0
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
        let distance = abs(anchor.transform.columns.3.z)

        // Detect emotion synchronously (only reads blendShapes, no frame retention)
        let blendShapesCopy = anchor.blendShapes

        // Overwrite synchronously — only ever 1 frame retained
        self.currentAnchor = anchor
        self.currentFrame = frame
        self.onARFrame?(frame)

        // Dispatch only scalars to main — closure does NOT capture frame/anchor
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            if self.gazeFilterX == nil {
                self.gazeFilterX = LowPassFilter(value: compensatedX)
                self.gazeFilterY = LowPassFilter(value: compensatedY)
            } else {
                self.gazeFilterX?.update(with: compensatedX)
                self.gazeFilterY?.update(with: compensatedY)
            }

            var newState = FaceState()
            newState.faceDetected = true
            newState.gazePoint = CGPoint(x: self.gazeFilterX?.value ?? adjustedX,
                                        y: self.gazeFilterY?.value ?? adjustedY)

            newState.headYaw = yaw
            newState.headPitch = pitch
            newState.headRoll = roll

            newState.headGesture = self.headGestureDetector.update(yaw: yaw, pitch: pitch, roll: roll)
            newState.emotion = self.emotionDetector.detectEmotion(from: blendShapesCopy, isSpeaking: false)
            newState.distanceFromCamera = distance

            let screenSize = self.cachedScreenSize
            let marginX = screenSize.width * 0.1
            let gazeX = self.gazeFilterX?.value ?? adjustedX
            let gazeY = self.gazeFilterY?.value ?? adjustedY
            let gazeInScreen = gazeX > marginX && gazeX < screenSize.width - marginX &&
                              gazeY > 0 && gazeY < screenSize.height
            // Head pose check: pitch -40°..50°, yaw ±30°
            let headPoseValid = (-0.7...0.87).contains(pitch) && (-0.52...0.52).contains(yaw)
            newState.isLookingAtScreen = gazeInScreen && headPoseValid

            newState.jawOpen = jawOpen
            newState.mouthClose = mouthClose
            newState.mouthFunnel = mouthFunnel
            newState.mouthPucker = mouthPucker
            newState.mouthLeft = mouthLeft
            newState.mouthRight = mouthRight
            newState.mouthStretchLeft = mouthStretchLeft
            newState.mouthStretchRight = mouthStretchRight
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

            // Append gaze trail at ~10fps
            let now = Date()
            if now.timeIntervalSince(self.lastTrailAppend) >= 0.1 {
                self.gazeTrail.append(newState.gazePoint)
                if self.gazeTrail.count > 100 { self.gazeTrail.removeFirst() }
                self.lastTrailAppend = now
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
