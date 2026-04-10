#if os(iOS)
import Foundation

public struct AudioState {
    public var volume: Float = 0       // 0.0 ~ 1.0 RMS
    public var isSpeaking: Bool = false

    public init(volume: Float = 0, isSpeaking: Bool = false) {
        self.volume = volume
        self.isSpeaking = isSpeaking
    }
}
#endif
