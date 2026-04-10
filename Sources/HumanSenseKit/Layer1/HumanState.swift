#if os(iOS)
import Foundation

public enum HumanActivity: String, CaseIterable {
    case absent           = "不在画面中"
    case eyesClosed       = "闭着眼睛"
    case distracted       = "看向别处"
    case listening        = "在看屏幕"
    case speakingToScreen = "对屏幕说话"
    case speakingToOther  = "对别处说话"

    public var emoji: String {
        switch self {
        case .absent:           return "👻"
        case .eyesClosed:       return "😴"
        case .distracted:       return "👀"
        case .listening:        return "👁"
        case .speakingToScreen: return "🗣"
        case .speakingToOther:  return "🗣"
        }
    }

    public var isSpeaking: Bool {
        self == .speakingToScreen || self == .speakingToOther
    }
}

public struct HumanState {
    public var activity: HumanActivity = .absent
    public var face: FaceState = FaceState()
    public var audio: AudioState = AudioState()
    public var hand: HandState = HandState()
    public var device: DeviceState = DeviceState()
    public var speech: SpeechState = SpeechState()

    public init() {}
}
#endif
