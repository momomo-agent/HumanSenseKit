import Foundation

enum HeadGesture: String {
    case none = "无"
    case nodding = "点头"
    case shaking = "摇头"
    case tiltingLeft = "左歪头"
    case tiltingRight = "右歪头"
}

/// Detects head gestures by analyzing pitch/yaw/roll history
class HeadGestureDetector {
    private var pitchHistory: [Float] = []
    private var yawHistory: [Float] = []
    private let windowSize = 20  // ~0.33s at 60fps
    private let minAmplitude: Float = 0.12  // ~7 degrees

    var currentGesture: HeadGesture = .none
    private var gestureCooldown: Int = 0

    func update(yaw: Float, pitch: Float, roll: Float) -> HeadGesture {
        pitchHistory.append(pitch)
        yawHistory.append(yaw)
        if pitchHistory.count > windowSize { pitchHistory.removeFirst() }
        if yawHistory.count > windowSize { yawHistory.removeFirst() }

        if gestureCooldown > 0 {
            gestureCooldown -= 1
            return currentGesture
        }

        // Detect nod: pitch oscillates (up-down or down-up)
        if detectOscillation(in: pitchHistory, minAmplitude: minAmplitude) {
            currentGesture = .nodding
            gestureCooldown = 30
            return .nodding
        }

        // Detect shake: yaw oscillates (left-right or right-left)
        if detectOscillation(in: yawHistory, minAmplitude: minAmplitude) {
            currentGesture = .shaking
            gestureCooldown = 30
            return .shaking
        }

        currentGesture = .none
        return .none
    }

    private func detectOscillation(in history: [Float], minAmplitude: Float) -> Bool {
        guard history.count >= windowSize else { return false }
        let mn = history.min()!
        let mx = history.max()!
        guard mx - mn > minAmplitude else { return false }

        // Check for direction reversal (at least one peak and one valley)
        var reversals = 0
        var lastDir: Float = 0
        for i in 1..<history.count {
            let dir = history[i] - history[i-1]
            if abs(dir) > 0.005 {
                if lastDir != 0 && dir * lastDir < 0 { reversals += 1 }
                lastDir = dir
            }
        }
        return reversals >= 2
    }
}
