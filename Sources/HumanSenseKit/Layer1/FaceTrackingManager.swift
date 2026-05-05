#if os(iOS)
import Foundation
import ARKit
import Combine

@MainActor
public class FaceTrackingManager: NSObject, ObservableObject {
    @Published public var faceState = FaceState()
    /// Latest ARFaceAnchor — updated synchronously on ARKit delegate thread.
    public nonisolated(unsafe) var currentAnchor: ARFaceAnchor?
    @Published public var gazeTrail: [CGPoint] = []
    @Published public var arSessionReady = false

    /// Real-time ARFrame callback for consumers (e.g. AvatarKit).
    /// Called synchronously on ARKit's delegate thread at ~60fps.
    /// WARNING: Do NOT retain the ARFrame - extract needed data immediately.
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

    /// When true, prefer a wide-angle front-camera video format for
    /// face tracking so the effective FOV is large enough to keep the
    /// user in frame even when they sit close to the device.
    ///
    /// On iPhone 15 Pro / 17 Pro, the front camera exposes a Center
    /// Stage ultra-wide sensor that `ARFaceTrackingConfiguration.
    /// supportedVideoFormats` lists with `captureDeviceType ==
    /// .builtInUltraWideCamera`. When available, picking that format
    /// roughly doubles the horizontal FOV vs the default narrow format.
    /// Falls back to the default ARKit-picked format on older devices
    /// or when no wide-angle format is advertised.
    ///
    /// Default `false` because some face tracking signals (e.g.
    /// `isLookingAtScreen` gaze estimation, head orientation) are
    /// calibrated against the narrow default format — switching to
    /// ultra-wide can bias those signals. Consumers can opt in via
    /// the public `HumanSenseKit.preferWideAngleCamera` setter.
    public var preferWideAngle: Bool = false

    public override init() {
        super.init()
    }

    /// Attach to an externally-owned ARSession.
    /// FaceTrackingManager becomes its delegate and configures face tracking.
    public func start(session: ARSession) {
        self.arSession = session
        self.hasSignaledReady = false
        self.arSessionReady = false
        guard ARFaceTrackingConfiguration.isSupported else {
            NSLog("[FaceTracking] ARFaceTracking NOT supported on this device")
            return
        }
        NSLog("[FaceTracking] Starting face tracking on shared ARSession")
        let config = ARFaceTrackingConfiguration()
        config.worldAlignment = .camera
        config.providesAudioData = false
        applyVideoFormat(to: config)
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    /// Select a preferred `ARVideoFormat` for the given configuration
    /// based on `preferWideAngle`. No-op when no wide-angle format is
    /// advertised (keeps ARKit's default).
    private func applyVideoFormat(to config: ARFaceTrackingConfiguration) {
        guard preferWideAngle else {
            NSLog("[FaceTracking] preferWideAngle=false, using ARKit default format")
            return
        }
        let formats = ARFaceTrackingConfiguration.supportedVideoFormats
        // iOS 16+ exposes captureDeviceType on ARVideoFormat. On older
        // OS this whole block simply won't find a match and we fall
        // back to the default.
        if #available(iOS 16.0, *) {
            // Prefer ultra-wide (Center Stage camera on 15 Pro / 17 Pro).
            if let wide = formats.first(where: { $0.captureDeviceType == .builtInUltraWideCamera }) {
                config.videoFormat = wide
                NSLog("[FaceTracking] Selected wide-angle format: %@ %dx%d @ %d fps",
                      String(describing: wide.captureDeviceType),
                      Int(wide.imageResolution.width), Int(wide.imageResolution.height),
                      wide.framesPerSecond)
                return
            }
        }
        NSLog("[FaceTracking] No ultra-wide format advertised, using ARKit default")
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
            // Call onARFrame callback but don't retain the frame
            self.onARFrame?(frame)
            if noFaceFrames >= noFaceThreshold {
                Task { @MainActor in
                    // Reset all face state to defaults when face disappears.
                    // Without this, UI reads stale values (jawOpen, gaze, etc.)
                    // and shows indicators as if user is still present.
                    self.faceState = FaceState()
                }
            }
            return
        }

        noFaceFrames = 0

        // Extract blendShapes even when !isTracked, so LipAudioCorrelator
        // gets real-time lip movement data for speech detection.
        // Only skip full state update if untracked for too long.
        let isTracked = anchor.isTracked
        if !isTracked {
            untrackedFrames += 1
            if untrackedFrames >= noFaceThreshold {
                Task { @MainActor in
                    self.faceState = FaceState()
                }
            }
            // Continue to extract blendShapes even when untracked
        } else {
            untrackedFrames = 0
        }

        // --- All extraction done synchronously on ARKit's delegate thread ---
        // No async queue = no closures retaining ARFrame

        let (yaw, pitch, roll) = extractHeadOrientation(from: anchor.transform)
        let orientation = cachedOrientation
        let size = cachedScreenSize

        // Eye-based gaze point (lookAtPoint projection)
        let lookAtVector = anchor.transform * SIMD4<Float>(anchor.lookAtPoint, 1)
        let lookPoint = frame.camera.projectPoint(
            SIMD3<Float>(lookAtVector.x, lookAtVector.y, lookAtVector.z),
            orientation: orientation,
            viewportSize: size
        )

        let adjustedX = size.width - lookPoint.x
        let adjustedY = size.height - lookPoint.y
        let compensatedX = adjustedX
        let compensatedY = adjustedY

        // Face-direction ray: project face forward vector tip for comparison
        let faceDistance = max(abs(anchor.transform.columns.3.z), 0.1)
        let faceForwardWorld = anchor.transform * SIMD4<Float>(0, 0, -Float(faceDistance), 1)
        let eyeLookPoint = frame.camera.projectPoint(
            SIMD3<Float>(faceForwardWorld.x, faceForwardWorld.y, faceForwardWorld.z),
            orientation: orientation,
            viewportSize: size
        )
        let eyeAdjustedX = size.width - eyeLookPoint.x
        let eyeAdjustedY = size.height - eyeLookPoint.y

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

        // Update anchor but don't retain frame to avoid memory warnings
        self.currentAnchor = anchor
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
            newState.gazePointEye = CGPoint(x: eyeAdjustedX, y: eyeAdjustedY)

            newState.headYaw = yaw
            newState.headPitch = pitch
            newState.headRoll = roll

            newState.headGesture = self.headGestureDetector.update(yaw: yaw, pitch: pitch, roll: roll)
            newState.emotion = self.emotionDetector.detectEmotion(from: blendShapesCopy, isSpeaking: false)
            newState.distanceFromCamera = distance

            let screenSize = self.cachedScreenSize
            let marginX = screenSize.width * 0.15
            let gazeX = self.gazeFilterX?.value ?? adjustedX
            let gazeY = self.gazeFilterY?.value ?? adjustedY

            // Fused gaze point: center + (face - eye) * scale
            let scaleX = distance > 0 ? 0.5 / Double(distance) : 1.0
            let scaleY = scaleX * (1.0 + abs(Double(pitch)) * 0.5)
            let centerX = Double(screenSize.width / 2)
            let centerY = Double(screenSize.height / 2)
            let diffX = Double(gazeX) - Double(eyeAdjustedX)
            let diffY = Double(gazeY) - Double(eyeAdjustedY)
            let fusedX = centerX + diffX * scaleX
            let fusedY = centerY + diffY * scaleY
            newState.gazePointFused = CGPoint(x: fusedX, y: fusedY)

            let fusedInScreen = fusedX > Double(marginX) && fusedX < Double(screenSize.width) - Double(marginX) &&
                                fusedY > 0 && fusedY < Double(screenSize.height)
            // Head pose check: pitch -40°..50°, yaw ±30°
            let headPoseValid = (-0.7...0.87).contains(pitch) && (-0.52...0.52).contains(yaw)
            newState.isLookingAtScreen = fusedInScreen && headPoseValid

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
        NSLog("[FaceTracking] ARSession interruption ended")
        // Only restart if this is still our active session.
        // When the app returns from background, HumanStateEngine.willEnterForeground
        // may have already called start() which creates a new session.
        // If we blindly restart the old session here, we'd overwrite the delegate
        // and the new session's face data would be lost.
        nonisolated(unsafe) let s = session
        Task { @MainActor in
            if self.arSession === s {
                NSLog("[FaceTracking] Restarting on same session (still active)")
                self.start(session: s)
            } else {
                NSLog("[FaceTracking] Ignoring interruption end — session replaced")
            }
        }
    }
}
#endif
