#if os(iOS)
import Foundation

public enum DevicePosture: String {
    case uprightStand = "竖放"
    case landscapeStand = "横放"
    case faceUp = "平躺"
    case faceDown = "盖着"
    case holdingWalking = "持握行走"
}

public enum DeviceOrientation: String {
    case portrait = "竖屏"
    case landscape = "横屏"
    case unknown = "未知"
}

public struct DeviceState {
    public var posture: DevicePosture = .uprightStand
    public var orientation: DeviceOrientation = .portrait
    public var isWalking: Bool = false
    public var isHolding: Bool = false
    /// Raw gravity vector from CMDeviceMotion.
    public var gravity: (x: Double, y: Double, z: Double) = (0, -1, 0)

    public init(posture: DevicePosture = .uprightStand, orientation: DeviceOrientation = .portrait, isWalking: Bool = false, isHolding: Bool = false) {
        self.posture = posture
        self.orientation = orientation
        self.isWalking = isWalking
        self.isHolding = isHolding
    }
}
#endif
