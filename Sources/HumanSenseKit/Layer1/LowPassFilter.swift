#if os(iOS)
import Foundation

public struct LowPassFilter {
    public var value: CGFloat
    public let alpha: CGFloat  // 0.85 recommended

    public init(value: CGFloat, alpha: CGFloat = 0.85) {
        self.value = value
        self.alpha = alpha
    }

    public mutating func update(with newValue: CGFloat) {
        value = alpha * value + (1.0 - alpha) * newValue
    }
}
#endif
