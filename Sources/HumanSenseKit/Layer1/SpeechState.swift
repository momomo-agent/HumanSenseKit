#if os(iOS)
import Foundation

public struct SpeechSegment: Identifiable {
    public let id: UUID
    public let text: String
    public let isToScreen: Bool
    public let sentenceStartedLookingAtScreen: Bool
    /// true = user's own speech (mouth moving + audio), false = ambient/other people
    public let isFromUser: Bool
    /// true = sentence has been finalized (silence gap detected), text won't change anymore
    public let isFinal: Bool

    public init(id: UUID = UUID(), text: String, isToScreen: Bool, sentenceStartedLookingAtScreen: Bool, isFromUser: Bool = false, isFinal: Bool = false) {
        self.id = id
        self.text = text
        self.isToScreen = isToScreen
        self.sentenceStartedLookingAtScreen = sentenceStartedLookingAtScreen
        self.isFromUser = isFromUser
        self.isFinal = isFinal
    }
}

public struct SpeechState {
    public var segments: [SpeechSegment] = []
    public var isListening: Bool = false

    public init(segments: [SpeechSegment] = [], isListening: Bool = false) {
        self.segments = segments
        self.isListening = isListening
    }
}
#endif
