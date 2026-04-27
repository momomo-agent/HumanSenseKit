#if os(iOS)
import Foundation
import Combine

/// Layer 2: High-level state queries
@MainActor
@Observable
public class HumanSenseState {
    /// The underlying raw sensor state snapshot.
    public private(set) var rawState: HumanState

    private let engine: HumanStateEngine

    /// Creates a new state wrapper around the given engine.
    /// - Parameter engine: The engine that produces raw human state.
    public init(engine: HumanStateEngine) {
        self.engine = engine
        self.rawState = engine.humanState
    }

    /// Call from a timer or frame callback to sync state
    public func update() {
        rawState = engine.humanState
    }

    // MARK: - High-level queries

    /// Whether a face is currently detected in the camera frame.
    public var isPresent: Bool { rawState.face.faceDetected }
    /// Whether the user is looking at the screen.
    public var isLookingAtScreen: Bool { rawState.face.isLookingAtScreen }
    /// Whether the user is currently speaking.
    public var isSpeaking: Bool { rawState.audio.isSpeaking }
    /// Whether STT has active (non-finalized) text being recognized.
    /// NOTE: `SpeechSegment.text` is now reconstructor-filtered (HSK 4.9.35+),
    /// so this is already "user-only" text — segments not attributed to the
    /// user have empty `.text`. Combined with `isFromUser`, this is the
    /// correct signal for "the user is actively saying something to the
    /// device right now".
    public var hasActiveSpeech: Bool {
        rawState.speech.segments.contains { seg in
            !seg.isFinal && seg.isFromUser && !seg.text.isEmpty
        }
    }
    /// Whether the user is speaking to the device.
    /// Gated ONLY by (a) looking at screen and (b) live STT user-attributed
    /// text. We intentionally DO NOT AND with `isSpeaking` (audio VAD)
    /// because that signal is a crude RMS-above-threshold check that fires
    /// on any ambient sound (other people, TV, keyboard, fans) and also
    /// has a 0.5s silence-delay latch that keeps it true well after the
    /// user stopped. `hasActiveSpeech` already requires a live segment that
    /// the reconstructor attributed to the user (jaw + gaze + head + STT
    /// tokens), which is a much stronger and cleaner signal for "the user
    /// is talking to the device right now".
    public var isSpeakingToDevice: Bool { isLookingAtScreen && hasActiveSpeech }
    /// Whether both eyes are closed.
    public var eyesClosed: Bool { rawState.face.eyesClosed }
    /// Distance from the user's face to the camera in meters.
    public var distanceFromCamera: Float { rawState.face.distanceFromCamera }
    /// Human-readable label for the face-to-camera distance.
    public var distanceLabel: String { rawState.face.distanceLabel }

    // Head gestures
    /// Whether the user is nodding.
    public var isNodding: Bool { rawState.face.headGesture == .nodding }
    /// Whether the user is shaking their head.
    public var isShakingHead: Bool { rawState.face.headGesture == .shaking }
    /// The currently detected head gesture.
    public var currentHeadGesture: HeadGesture { rawState.face.headGesture }

    // Hand gestures
    /// Whether a hand is currently detected.
    public var handDetected: Bool { rawState.hand.detected }
    /// The currently recognized hand gesture.
    public var currentHandGesture: HandGesture { rawState.hand.gesture }
    /// Whether the detected hand is the left hand.
    public var isLeftHand: Bool { rawState.hand.isLeftHand }
    /// Whether the detected hand is the right hand.
    public var isRightHand: Bool { !rawState.hand.isLeftHand && rawState.hand.detected }

    // Gaze
    /// Estimated screen-space gaze point.
    public var gazePoint: CGPoint { rawState.face.gazePoint }

    // Head orientation (radians)
    /// Head yaw angle in radians (left/right rotation).
    public var headYaw: Float { rawState.face.headYaw }
    /// Head pitch angle in radians (up/down tilt).
    public var headPitch: Float { rawState.face.headPitch }
    /// Head roll angle in radians (side tilt).
    public var headRoll: Float { rawState.face.headRoll }
}
#endif
