import Foundation
import ARKit

struct FaceState {
    // Gaze
    var gazePoint: CGPoint = .zero
    var isLookingAtScreen: Bool = false

    // Head orientation (radians)
    var headYaw: Float = 0    // left/right
    var headPitch: Float = 0  // up/down
    var headRoll: Float = 0   // tilt

    // Distance from camera (meters)
    var distanceFromCamera: Float = 0

    // Mouth
    var jawOpen: Float = 0
    var mouthClose: Float = 0

    // Eyes
    var eyeBlinkLeft: Float = 0
    var eyeBlinkRight: Float = 0
    var eyeLookInLeft: Float = 0
    var eyeLookOutLeft: Float = 0
    var eyeLookUpLeft: Float = 0
    var eyeLookDownLeft: Float = 0
    var eyeLookInRight: Float = 0
    var eyeLookOutRight: Float = 0
    var eyeLookUpRight: Float = 0
    var eyeLookDownRight: Float = 0

    // Derived
    var gazeH: Float { ((eyeLookInLeft - eyeLookOutLeft) + (eyeLookOutRight - eyeLookInRight)) / 2 }
    var gazeV: Float { ((eyeLookDownLeft - eyeLookUpLeft) + (eyeLookDownRight - eyeLookUpRight)) / 2 }
    var eyesClosed: Bool { eyeBlinkLeft > 0.8 && eyeBlinkRight > 0.8 }
    var faceDetected: Bool = false

    // Head gesture
    var headGesture: HeadGesture = .none

    // Distance label
    var distanceLabel: String {
        switch distanceFromCamera {
        case 0: return "未知"
        case ..<0.3: return "很近"
        case ..<0.5: return "近"
        case ..<0.8: return "适中"
        case ..<1.2: return "远"
        default: return "很远"
        }
    }
}
