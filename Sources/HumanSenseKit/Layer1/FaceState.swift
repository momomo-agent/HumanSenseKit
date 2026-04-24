#if os(iOS)
import Foundation
import ARKit

public enum HeadOrientation {
    case forward, left, right

    public static let threshold: Float = 0.3  // ~17 degrees

    public init(yaw: Float) {
        if yaw > Self.threshold { self = .left }
        else if yaw < -Self.threshold { self = .right }
        else { self = .forward }
    }

    public var isFacingForward: Bool { self == .forward }
}

public struct FaceState {
    // Gaze
    public var gazePoint: CGPoint = .zero       // face-direction ray cast
    public var gazePointEye: CGPoint = .zero    // eye lookAtPoint projection
    public var isLookingAtScreen: Bool = false

    // Head orientation (radians)
    public var headYaw: Float = 0    // left/right
    public var headPitch: Float = 0  // up/down
    public var headRoll: Float = 0   // tilt

    // Distance from camera (meters)
    public var distanceFromCamera: Float = 0

    // Mouth
    public var jawOpen: Float = 0
    public var mouthClose: Float = 0
    public var mouthFunnel: Float = 0
    public var mouthPucker: Float = 0
    public var mouthLeft: Float = 0
    public var mouthRight: Float = 0
    public var mouthStretchLeft: Float = 0
    public var mouthStretchRight: Float = 0

    // Eyes
    public var eyeBlinkLeft: Float = 0
    public var eyeBlinkRight: Float = 0
    public var eyeLookInLeft: Float = 0
    public var eyeLookOutLeft: Float = 0
    public var eyeLookUpLeft: Float = 0
    public var eyeLookDownLeft: Float = 0
    public var eyeLookInRight: Float = 0
    public var eyeLookOutRight: Float = 0
    public var eyeLookUpRight: Float = 0
    public var eyeLookDownRight: Float = 0

    // Derived
    public var gazeH: Float { ((eyeLookInLeft - eyeLookOutLeft) + (eyeLookOutRight - eyeLookInRight)) / 2 }
    public var gazeV: Float { ((eyeLookDownLeft - eyeLookUpLeft) + (eyeLookDownRight - eyeLookUpRight)) / 2 }
    public var eyesClosed: Bool { eyeBlinkLeft > 0.8 && eyeBlinkRight > 0.8 }
    public var faceDetected: Bool = false

    public var headOrientation: HeadOrientation { HeadOrientation(yaw: headYaw) }

    // Head gesture
    public var headGesture: HeadGesture = .none

    // Emotion
    public var emotion: Emotion = .neutral

    // Distance label
    public var distanceLabel: String {
        switch distanceFromCamera {
        case 0: return "未知"
        case ..<0.3: return "很近"
        case ..<0.5: return "近"
        case ..<0.8: return "适中"
        case ..<1.2: return "远"
        default: return "很远"
        }
    }

    public init() {}
}
#endif
