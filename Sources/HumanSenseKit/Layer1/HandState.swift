#if os(iOS)
import Foundation
import Vision

// MARK: - Gesture Description (declarative, like fingerpose)

/// A constraint on one finger's curl or direction
public struct FingerConstraint {
    public let finger: Finger
    public let curl: FingerCurl?
    public let direction: FingerDirection?
    public let weight: Double  // 0..1, how important this constraint is

    public init(_ finger: Finger, curl: FingerCurl? = nil, direction: FingerDirection? = nil, weight: Double = 1.0) {
        self.finger = finger
        self.curl = curl
        self.direction = direction
        self.weight = weight
    }
}

/// Declarative gesture definition
public struct GestureDescription {
    public let gesture: HandGesture
    public let constraints: [FingerConstraint]
    /// Optional: require thumb-to-finger tip distance < threshold
    public let pinchFinger: Finger?
    public let pinchThreshold: Double

    public init(_ gesture: HandGesture, constraints: [FingerConstraint],
         pinchFinger: Finger? = nil, pinchThreshold: Double = 0.08) {
        self.gesture = gesture
        self.constraints = constraints
        self.pinchFinger = pinchFinger
        self.pinchThreshold = pinchThreshold
    }

    /// Score 0..10 for how well finger data matches this gesture
    public func score(fingers: [Finger: FingerData], analyzer: FingerAnalyzer, observation: VNHumanHandPoseObservation) -> Double {
        var totalWeight = 0.0
        var matchedWeight = 0.0

        for c in constraints {
            guard let data = fingers[c.finger] else { return 0 }
            totalWeight += c.weight

            var matched = true
            if let expectedCurl = c.curl {
                if data.curl != expectedCurl {
                    // Allow adjacent curl with partial credit
                    let curlDist = abs(data.curl.ordinal - expectedCurl.ordinal)
                    if curlDist == 1 {
                        matchedWeight += c.weight * 0.5  // half credit for adjacent
                        continue
                    }
                    matched = false
                }
            }
            if let expectedDir = c.direction, matched {
                if data.direction != expectedDir {
                    // Check if adjacent direction
                    if areAdjacentDirections(data.direction, expectedDir) {
                        matchedWeight += c.weight * 0.8
                        continue
                    }
                    matched = false
                }
            }

            if matched {
                matchedWeight += c.weight
            }
        }

        guard totalWeight > 0 else { return 0 }
        var score = (matchedWeight / totalWeight) * 10.0

        // Pinch check
        if let pinchFinger = pinchFinger {
            let jointName: VNHumanHandPoseObservation.JointName
            switch pinchFinger {
            case .index: jointName = .indexTip
            case .middle: jointName = .middleTip
            case .ring: jointName = .ringTip
            case .pinky: jointName = .littleTip
            case .thumb: jointName = .thumbTip
            }
            if let dist = analyzer.tipDistance(observation, .thumbTip, jointName) {
                if dist > pinchThreshold {
                    score *= 0.3  // Heavy penalty if pinch not detected
                }
            }
        }

        return score
    }
}

// MARK: - Direction adjacency

private func areAdjacentDirections(_ a: FingerDirection, _ b: FingerDirection) -> Bool {
    let order: [FingerDirection] = [
        .verticalUp, .diagonalUpRight, .horizontalRight, .diagonalDownRight,
        .verticalDown, .diagonalDownLeft, .horizontalLeft, .diagonalUpLeft
    ]
    guard let ia = order.firstIndex(of: a), let ib = order.firstIndex(of: b) else { return false }
    let diff = abs(ia - ib)
    return diff == 1 || diff == order.count - 1
}

extension FingerCurl {
    public var ordinal: Int {
        switch self {
        case .noCurl: return 0
        case .halfCurl: return 1
        case .fullCurl: return 2
        }
    }
}

// MARK: - Hand Gesture enum

public enum HandGesture: String, Equatable {
    case none = "无手势"
    case thumbsUp = "👍 点赞"
    case peace = "✌️ 胜利"
    case openPalm = "🖐 张开手掌"
    case fist = "✊ 握拳"
    case pointing = "☝️ 指向"
    case ok = "👌 OK"
    case rock = "🤘 摇滚"
    case one = "1️⃣ 数字1"
    case two = "2️⃣ 数字2"
    case three = "3️⃣ 数字3"
    case four = "4️⃣ 数字4"
    case five = "5️⃣ 数字5"
}

// MARK: - Gesture Library

public struct GestureLibrary {
    public static let analyzer = FingerAnalyzer()

    public static let gestures: [GestureDescription] = [
        // 👍 Thumbs up: thumb straight, all others curled
        GestureDescription(.thumbsUp, constraints: [
            FingerConstraint(.thumb, curl: .noCurl),
            FingerConstraint(.index, curl: .fullCurl),
            FingerConstraint(.middle, curl: .fullCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        // ✌️ Peace / Victory: index + middle straight, others curled
        GestureDescription(.peace, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl, weight: 0.6),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .noCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        // 🖐 Open palm: all fingers straight
        GestureDescription(.openPalm, constraints: [
            FingerConstraint(.thumb, curl: .noCurl),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .noCurl),
            FingerConstraint(.ring, curl: .noCurl),
            FingerConstraint(.pinky, curl: .noCurl),
        ]),

        // ✊ Fist: all fingers curled
        GestureDescription(.fist, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl),
            FingerConstraint(.index, curl: .fullCurl),
            FingerConstraint(.middle, curl: .fullCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        // ☝️ Pointing: only index straight
        GestureDescription(.pointing, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl, weight: 0.6),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .fullCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        // 👌 OK: thumb + index pinch, others straight
        GestureDescription(.ok, constraints: [
            FingerConstraint(.thumb, curl: .halfCurl, weight: 0.5),
            FingerConstraint(.index, curl: .halfCurl, weight: 0.5),
            FingerConstraint(.middle, curl: .noCurl),
            FingerConstraint(.ring, curl: .noCurl),
            FingerConstraint(.pinky, curl: .noCurl),
        ], pinchFinger: .index, pinchThreshold: 0.08),

        // 🤘 Rock: index + pinky straight, middle + ring curled
        GestureDescription(.rock, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl, weight: 0.5),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .fullCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .noCurl),
        ]),

        // Number gestures
        GestureDescription(.one, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl, weight: 0.6),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .fullCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        GestureDescription(.three, constraints: [
            FingerConstraint(.thumb, curl: .noCurl),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .noCurl),
            FingerConstraint(.ring, curl: .fullCurl),
            FingerConstraint(.pinky, curl: .fullCurl),
        ]),

        GestureDescription(.four, constraints: [
            FingerConstraint(.thumb, curl: .fullCurl),
            FingerConstraint(.index, curl: .noCurl),
            FingerConstraint(.middle, curl: .noCurl),
            FingerConstraint(.ring, curl: .noCurl),
            FingerConstraint(.pinky, curl: .noCurl),
        ]),
    ]

    /// Match the best gesture from finger data. Returns .none if no gesture scores above threshold.
    public static func match(observation: VNHumanHandPoseObservation) -> HandGesture {
        guard let fingers = analyzer.analyze(observation) else { return .none }

        let threshold = 7.0  // Minimum score out of 10
        var bestGesture: HandGesture = .none
        var bestScore = 0.0

        for desc in gestures {
            let score = desc.score(fingers: fingers, analyzer: analyzer, observation: observation)
            if score > bestScore && score >= threshold {
                bestScore = score
                bestGesture = desc.gesture
            }
        }

        return bestGesture
    }
}

// MARK: - Hand State

public struct HandState {
    public var detected: Bool = false
    public var gesture: HandGesture = .none
    public var fingerData: [Finger: FingerData] = [:]
    public var handPoints: [CGPoint] = []
    public var isLeftHand: Bool = false
    public var handLabel: String {
        guard detected else { return "未检测到手" }
        return (isLeftHand ? "左手 " : "右手 ") + gesture.rawValue
    }

    public init(detected: Bool = false, gesture: HandGesture = .none, fingerData: [Finger: FingerData] = [:], handPoints: [CGPoint] = [], isLeftHand: Bool = false) {
        self.detected = detected
        self.gesture = gesture
        self.fingerData = fingerData
        self.handPoints = handPoints
        self.isLeftHand = isLeftHand
    }
}
#endif
