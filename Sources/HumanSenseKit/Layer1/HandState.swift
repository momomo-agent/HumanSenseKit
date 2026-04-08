import Foundation
import Vision

enum HandGesture: String {
    case none = "无手势"
    case thumbsUp = "👍 点赞"
    case peace = "✌️ 胜利"
    case openPalm = "🖐 张开手掌"
    case fist = "✊ 握拳"
    case pointing = "☝️ 指向"
    
    static func detect(from observation: VNHumanHandPoseObservation) -> HandGesture {
        guard let thumbTip = try? observation.recognizedPoint(.thumbTip),
              let indexTip = try? observation.recognizedPoint(.indexTip),
              let middleTip = try? observation.recognizedPoint(.middleTip),
              let ringTip = try? observation.recognizedPoint(.ringTip),
              let littleTip = try? observation.recognizedPoint(.littleTip),
              let wrist = try? observation.recognizedPoint(.wrist) else {
            return .none
        }
        
        // All points must have reasonable confidence
        guard thumbTip.confidence > 0.3 && indexTip.confidence > 0.3 &&
              middleTip.confidence > 0.3 && ringTip.confidence > 0.3 &&
              littleTip.confidence > 0.3 else {
            return .none
        }
        
        let thumbUp = thumbTip.location.y > wrist.location.y + 0.15
        let indexUp = indexTip.location.y > wrist.location.y + 0.1
        let middleUp = middleTip.location.y > wrist.location.y + 0.1
        let ringDown = ringTip.location.y < wrist.location.y + 0.05
        let littleDown = littleTip.location.y < wrist.location.y + 0.05
        
        // Thumbs up: thumb up, others down
        if thumbUp && !indexUp && !middleUp && ringDown && littleDown {
            return .thumbsUp
        }
        
        // Peace: index + middle up, others down
        if indexUp && middleUp && !thumbUp && ringDown && littleDown {
            return .peace
        }
        
        // Open palm: all fingers up
        if thumbUp && indexUp && middleUp &&
           ringTip.location.y > wrist.location.y &&
           littleTip.location.y > wrist.location.y {
            return .openPalm
        }
        
        // Fist: all fingers down
        if !thumbUp && !indexUp && !middleUp && ringDown && littleDown {
            return .fist
        }
        
        // Pointing: only index up
        if indexUp && !middleUp && !thumbUp && ringDown && littleDown {
            return .pointing
        }
        
        return .none
    }
}

struct HandState {
    var detected: Bool = false
    var gesture: HandGesture = .none
    var handPoints: [CGPoint] = []
    var isLeftHand: Bool = false
    var handLabel: String {
        guard detected else { return "未检测到手" }
        return (isLeftHand ? "左手 " : "右手 ") + gesture.rawValue
    }
}
