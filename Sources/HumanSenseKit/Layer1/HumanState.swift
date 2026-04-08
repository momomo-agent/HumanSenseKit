import Foundation

enum HumanActivity: String, CaseIterable {
    case absent       = "不在画面中"
    case eyesClosed   = "闭着眼睛"
    case distracted   = "看向别处"
    case listening    = "在看屏幕"
    case speaking     = "正在说话"

    var emoji: String {
        switch self {
        case .absent:     return "👻"
        case .eyesClosed: return "😴"
        case .distracted: return "👀"
        case .listening:  return "👁"
        case .speaking:   return "🗣"
        }
    }

    var color: String {
        switch self {
        case .absent:     return "gray"
        case .eyesClosed: return "purple"
        case .distracted: return "orange"
        case .listening:  return "green"
        case .speaking:   return "blue"
        }
    }
}

struct HumanState {
    var activity: HumanActivity = .absent
    var face: FaceState = FaceState()
    var audio: AudioState = AudioState()
    var hand: HandState = HandState()
}
