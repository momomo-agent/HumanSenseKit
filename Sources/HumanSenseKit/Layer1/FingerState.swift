#if os(iOS)
import Foundation
import Vision

// MARK: - Finger Curl & Direction (fingerpose approach)

/// How curled a finger is, based on joint angles
public enum FingerCurl: String, CaseIterable {
    case noCurl = "伸直"
    case halfCurl = "半弯"
    case fullCurl = "全弯"
}

/// Which direction a finger points, relative to the hand's own frame
public enum FingerDirection: String, CaseIterable {
    case verticalUp = "↑"
    case verticalDown = "↓"
    case horizontalLeft = "←"
    case horizontalRight = "→"
    case diagonalUpLeft = "↖"
    case diagonalUpRight = "↗"
    case diagonalDownLeft = "↙"
    case diagonalDownRight = "↘"
}

public enum Finger: Int, CaseIterable {
    case thumb = 0
    case index = 1
    case middle = 2
    case ring = 3
    case pinky = 4
}

public struct FingerData {
    public let curl: FingerCurl
    public let direction: FingerDirection

    public init(curl: FingerCurl, direction: FingerDirection) {
        self.curl = curl
        self.direction = direction
    }
}

// MARK: - Geometry helpers

/// Angle at vertex B in triangle A-B-C (radians)
private func angleBetween(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double {
    let ba = CGPoint(x: a.x - b.x, y: a.y - b.y)
    let bc = CGPoint(x: c.x - b.x, y: c.y - b.y)
    let dot = ba.x * bc.x + ba.y * bc.y
    let magBA = sqrt(ba.x * ba.x + ba.y * ba.y)
    let magBC = sqrt(bc.x * bc.x + bc.y * bc.y)
    guard magBA > 0 && magBC > 0 else { return .pi }
    let cosAngle = max(-1, min(1, dot / (magBA * magBC)))
    return acos(cosAngle)
}

/// Direction from point A to point B
private func direction(from a: CGPoint, to b: CGPoint) -> FingerDirection {
    let dx = b.x - a.x
    let dy = b.y - a.y  // Vision: y increases upward
    let angle = atan2(dy, dx) // radians, 0 = right, π/2 = up
    let deg = angle * 180 / .pi

    // Map angle to 8 directions
    // Note: Vision coordinate system has y pointing up
    switch deg {
    case 67.5..<112.5:   return .verticalUp
    case 22.5..<67.5:    return .diagonalUpRight
    case -22.5..<22.5:   return .horizontalRight
    case -67.5 ..< -22.5: return .diagonalDownRight
    case -112.5 ..< -67.5: return .verticalDown
    case -157.5 ..< -112.5: return .diagonalDownLeft
    default:
        // covers 112.5..180 and -180..-157.5
        if deg >= 112.5 || deg < -157.5 { return .horizontalLeft }
        if deg >= 112.5 && deg < 157.5 { return .diagonalUpLeft }
        return .horizontalLeft
    }
}

// MARK: - Finger analysis from Vision joints

public struct FingerAnalyzer {

    private let minConfidence: Float = 0.3

    public init() {}

    /// Analyze all five fingers from a hand pose observation
    public func analyze(_ observation: VNHumanHandPoseObservation) -> [Finger: FingerData]? {
        var result: [Finger: FingerData] = [:]

        for finger in Finger.allCases {
            if finger == .thumb {
                if let data = analyzeThumb(observation) {
                    result[finger] = data
                }
            } else {
                if let data = analyzeFinger(observation, finger: finger) {
                    result[finger] = data
                }
            }
        }

        return result.isEmpty ? nil : result
    }

    private func analyzeFinger(_ observation: VNHumanHandPoseObservation, finger: Finger) -> FingerData? {
        let jointGroup: VNHumanHandPoseObservation.JointsGroupName
        switch finger {
        case .index: jointGroup = .indexFinger
        case .middle: jointGroup = .middleFinger
        case .ring: jointGroup = .ringFinger
        case .pinky: jointGroup = .littleFinger
        case .thumb: return nil
        }

        guard let points = try? observation.recognizedPoints(jointGroup) else { return nil }

        // Get MCP, PIP, DIP, TIP
        let keys = points.keys.sorted { $0.rawValue.rawValue < $1.rawValue.rawValue }
        let highConfidence = points.values.filter { $0.confidence > minConfidence }
        guard highConfidence.count >= 4 else { return nil }

        let sortedPoints = keys.compactMap { points[$0] }.filter { $0.confidence > minConfidence }
        guard sortedPoints.count >= 4 else { return nil }

        let mcp = sortedPoints[0].location
        let pip = sortedPoints[1].location
        let dip = sortedPoints[2].location
        let tip = sortedPoints[3].location

        // Calculate curl from joint angles
        let pipAngle = angleBetween(mcp, pip, dip)
        let dipAngle = angleBetween(pip, dip, tip)
        let avgAngle = (pipAngle + dipAngle) / 2.0

        let curl: FingerCurl
        if avgAngle > 2.6 {
            curl = .noCurl
        } else if avgAngle > 1.8 {
            curl = .halfCurl
        } else {
            curl = .fullCurl
        }

        let dir = direction(from: mcp, to: tip)

        return FingerData(curl: curl, direction: dir)
    }

    private func analyzeThumb(_ observation: VNHumanHandPoseObservation) -> FingerData? {
        guard let pCMC = try? observation.recognizedPoint(.thumbCMC),
              let pMP = try? observation.recognizedPoint(.thumbMP),
              let pIP = try? observation.recognizedPoint(.thumbIP),
              let pTIP = try? observation.recognizedPoint(.thumbTip),
              pCMC.confidence > minConfidence,
              pMP.confidence > minConfidence,
              pIP.confidence > minConfidence,
              pTIP.confidence > minConfidence else {
            return nil
        }

        let cmcLoc = pCMC.location
        let mpLoc = pMP.location
        let ipLoc = pIP.location
        let tipLoc = pTIP.location

        // Thumb has fewer degrees of freedom — use IP angle primarily
        let ipAngle = angleBetween(mpLoc, ipLoc, tipLoc)
        let mpAngle = angleBetween(cmcLoc, mpLoc, ipLoc)
        let avgAngle = (ipAngle + mpAngle) / 2.0

        let curl: FingerCurl
        if avgAngle > 2.4 {        // Thumb can't straighten as much
            curl = .noCurl
        } else if avgAngle > 1.6 {
            curl = .halfCurl
        } else {
            curl = .fullCurl
        }

        let dir = direction(from: cmcLoc, to: tipLoc)

        return FingerData(curl: curl, direction: dir)
    }

    /// Distance between thumb tip and another fingertip (for pinch/OK detection)
    public func tipDistance(_ observation: VNHumanHandPoseObservation,
                     _ a: VNHumanHandPoseObservation.JointName,
                     _ b: VNHumanHandPoseObservation.JointName) -> Double? {
        guard let pA = try? observation.recognizedPoint(a),
              let pB = try? observation.recognizedPoint(b),
              pA.confidence > minConfidence, pB.confidence > minConfidence else {
            return nil
        }
        return hypot(pA.location.x - pB.location.x, pA.location.y - pB.location.y)
    }
}
#endif
