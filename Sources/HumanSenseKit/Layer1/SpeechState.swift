#if os(iOS)
import Foundation

public struct SpeechSegment: Identifiable {
    public let id: UUID
    public let text: String
    public let isToScreen: Bool
    public let sentenceStartedLookingAtScreen: Bool
    /// Exponentially-weighted "talking-to-AI" score in [0, 1].
    /// Early characters dominate (λ=0.5 → first 3-5 chars carry ~95% of weight),
    /// but later chars still contribute. 1.0 = whole sentence spoken at screen,
    /// 0.0 = never at screen, in-between = partially addressed.
    public let speakingToAIScore: Float
    /// true = user's own speech (mouth moving + audio), false = ambient/other people
    public let isFromUser: Bool
    /// true = sentence has been finalized (silence gap detected), text won't change anymore
    public let isFinal: Bool

    /// Debug snapshot of signals at the time this segment was created/updated
    public struct SignalSnapshot {
        public let mouthMoving: Bool
        public let audioActive: Bool
        public let gazeOnScreen: Bool
        public let headForward: Bool
        public let lipCorrelation: Float
        public let lipCorrelated: Bool
        public let activity: String
        /// Best cross-correlation offset in frames
        public let bestOffset: Int
        /// Waveform snapshot at the time of segment creation
        public let waveform: [LipAudioCorrelator.SamplePoint]

        public init(mouthMoving: Bool = false, audioActive: Bool = false, gazeOnScreen: Bool = false,
                    headForward: Bool = false, lipCorrelation: Float = 0, lipCorrelated: Bool = false,
                    activity: String = "", bestOffset: Int = 0,
                    waveform: [LipAudioCorrelator.SamplePoint] = []) {
            self.mouthMoving = mouthMoving
            self.audioActive = audioActive
            self.gazeOnScreen = gazeOnScreen
            self.headForward = headForward
            self.lipCorrelation = lipCorrelation
            self.lipCorrelated = lipCorrelated
            self.activity = activity
            self.bestOffset = bestOffset
            self.waveform = waveform
        }
    }
    public let signals: SignalSnapshot

    public init(id: UUID = UUID(), text: String, isToScreen: Bool, sentenceStartedLookingAtScreen: Bool,
                speakingToAIScore: Float? = nil,
                isFromUser: Bool = false, isFinal: Bool = false, signals: SignalSnapshot = SignalSnapshot()) {
        self.id = id
        self.text = text
        self.isToScreen = isToScreen
        self.sentenceStartedLookingAtScreen = sentenceStartedLookingAtScreen
        // Backward-compat default: mirror the bool when no explicit score is supplied.
        self.speakingToAIScore = speakingToAIScore ?? (sentenceStartedLookingAtScreen ? 1.0 : 0.0)
        self.isFromUser = isFromUser
        self.isFinal = isFinal
        self.signals = signals
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
