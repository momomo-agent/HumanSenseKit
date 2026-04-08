import Foundation

struct LowPassFilter {
    var value: CGFloat
    let alpha: CGFloat  // 0.85 recommended
    
    init(value: CGFloat, alpha: CGFloat = 0.85) {
        self.value = value
        self.alpha = alpha
    }
    
    mutating func update(with newValue: CGFloat) {
        value = alpha * value + (1.0 - alpha) * newValue
    }
}
