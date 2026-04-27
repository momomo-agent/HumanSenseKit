#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// STTManager — thin orchestrator composing three layers:
///   Layer 0: AudioEngineManager  (audio lifecycle)
///   Layer 1: SpeechRecognitionBackend  (SpeechAnalyzer or SFSpeechRecognizer)
///   Layer 2: SentenceBuilder  (sentence model + gaze tracking)
@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    @Published public var lastError: String?
    /// Current speaker label from diarization (nil if unavailable).
    @Published public var currentSpeakerLabel: String?

    /// Which speech recognition engine to use.
    public enum BackendType: Sendable {
        /// iOS 26 SpeechAnalyzer (default).
        case speechAnalyzer
        /// Legacy SFSpeechRecognizer (fallback).
        case sfSpeechRecognizer
    }

    /// When true, audio buffers are discarded and STT is effectively paused.
    /// Use this to suppress recognition when the user is not facing the screen.
    public var isMuted: Bool = false

    @Published public var speechDetected: Bool = false

    /// Contextual strings to improve speech recognition accuracy.
    /// Set before calling start(). Examples: app names, person names, technical terms.
    public var contextualStrings: [String] = [] {
        didSet {
            if let backend = speech as? SpeechAnalyzerBackend {
                backend.contextualStrings = contextualStrings
            }
        }
    }

    /// Wall-clock time when the audio stream started (when start() was called).
    /// Use this to convert audio-relative CMTime offsets to system time:
    /// systemTime = audioStreamStartTime + audioOffset
    public private(set) var audioStreamStartTime: Date?

    /// Called when a sentence is finalized — for debug logging.
    public var onSentenceFinalized: ((_ text: String) -> Void)?

    /// Called whenever new token-level transcription data arrives (volatile or final).
    /// Each token carries its audio time range (seconds, relative to audio stream start).
    /// Combine with `audioStreamStartTime` to align with wall-clock signals.
    public var onTokens: ((_ tokens: [SpeechToken], _ isFinal: Bool) -> Void)?

    /// Set a closure to capture signal snapshots for debug display on each segment.
    public var captureSignals: (() -> SpeechSegment.SignalSnapshot)? {
        get { builder.captureSignals }
        set { builder.captureSignals = newValue }
    }

    // --- Sub-components ---
    private let audio = AudioEngineManager()
    private let speech: any SpeechRecognitionBackend
    private let builder = SentenceBuilder()

    /// Token-level attribution engine. Queries correlator Pearson history
    /// to assign per-token isFromUser confidence.
    public let tokenAttributor = TokenAttributor()

    /// Newer user-sentence reconstruction layer. Supersedes `tokenAttributor`
    /// + `SentenceBuilder` for apps that just want `userSentence` (the live
    /// or most-recently-finalized text attributed to the user). Runs in
    /// parallel with the legacy attributor during the transition period.
    /// HumanStateEngine feeds it per-frame jaw/vol/gaze/head samples and the
    /// STT onTokens stream is routed here automatically.
    public let userSentenceReconstructor = UserSentenceReconstructor()

    /// The audio engine. Can be replaced before calling start().
    public var audioEngine: AVAudioEngine {
        get { audio.audioEngine }
        set { audio.audioEngine = newValue }
    }

    /// When true, STTManager will NOT manage the audio engine lifecycle.
    public var usesExternalEngine: Bool {
        get { audio.usesExternalEngine }
        set { audio.usesExternalEngine = newValue }
    }

    /// AudioDetectionManager receives buffers from our shared tap.
    public weak var audioDetectionManager: AudioDetectionManager?

    // --- External inputs (set by Engine) ---
    public var isLookingAtScreen: Bool = false {
        didSet { builder.isLookingAtScreen = isLookingAtScreen }
    }
    public var isSpeaking: Bool = false {
        didSet { builder.isSpeaking = isSpeaking }
    }
    /// Onset-weighted gaze score for the current utterance, fed by
    /// HumanStateEngine every frame. SentenceBuilder reads this directly
    /// so the per-sentence score reflects audio-frame-rate sampling, not
    /// the few times STT happens to emit volatile text.
    @Published public var onsetGazeScore: Float = 0 {
        didSet { builder.onsetGazeScore = onsetGazeScore }
    }
    @Published public var onsetFrameCount: Int = 0
    @Published public var onsetLookAtCount: Int = 0
    @Published public var onsetCorrCount: Int = 0

    public func captureSpeechStartState() {}

    // MARK: - Init

    /// Create an STTManager with the specified backend.
    /// Defaults to `.speechAnalyzer` (iOS 26).
    public init(backend: BackendType = .speechAnalyzer) {
        switch backend {
        case .speechAnalyzer:
            self.speech = SpeechAnalyzerBackend()
        case .sfSpeechRecognizer:
            self.speech = SFSpeechBackend()
        }
        super.init()
    }

    // MARK: - Lifecycle

    public func start() {
        print("[STT] start()")
        speech.authorize { [weak self] authorized in
            guard authorized else {
                print("[STT] Not authorized")
                return
            }
            Task { @MainActor in
                self?.lastError = nil
                self?.wireUp()
                self?.audio.start()
                self?.speech.startTask()
                // Stamp audioStreamStartTime right after startTask — the analyzer
                // timeline begins from the first buffer it consumes, which is
                // approximately now (within a few ms), not when STT.start() was
                // invoked (that can be seconds earlier due to auth/model download).
                self?.audioStreamStartTime = Date()
                self?.builder.resetActive()
                self?.isListening = true
            }
        }
    }

    public func stop() {
        speech.stop()
        audio.stop()
        builder.finalizeAndReset()
        rebuildSegments()
        isListening = false
    }

    /// Notify STTManager that the underlying audio engine was restarted.
    ///
    /// When an AVAudioEngine is stopped and re-started (e.g. app returned
    /// from background, AVAudioSession interruption ended, engine config
    /// change forced a restart), its sample timeline resets to 0. Any
    /// token audioTimeRange emitted AFTER that point is relative to the
    /// NEW stream start, not the original one. If `audioStreamStartTime`
    /// is not updated, token wall-clock times are computed from a stale
    /// base and land far outside the sample ring buffer's wall-clock
    /// range, producing `sampleCount=0` rows in the reconstructor — all
    /// jaw/vol/gaze/head signals read as zero and the token is wrongly
    /// judged non-user.
    ///
    /// For `usesExternalEngine=true` setups, call this from the host
    /// app's audio engine restart path (e.g. after `engine.start()` on
    /// interruption end / config change / scene activation).
    ///
    /// Safe to call repeatedly; `onFirstBuffer` will replace the coarse
    /// stamp set here with the precise time the analyzer consumes its
    /// next buffer.
    public func notifyAudioStreamRestarted() {
        audioStreamStartTime = Date()
        tokenAttributor.audioStreamStartTime = Date()
        // Clear the reconstructor state — old token rows reference the
        // old stream timeline and would poison range queries under the
        // new timeline.
        userSentenceReconstructor.clear()
        speech.startTask()
        builder.resetActive()
    }

    public func clearSegments() {
        builder.clearAll()
        userSentenceReconstructor.clear()
        rebuildSegments()
    }

    // MARK: - Wiring

    private func wireUp() {
        // Capture sub-components directly — avoid accessing @MainActor self from audio thread
        let speech = self.speech
        weak var audioDetection = self.audioDetectionManager

        // Layer 0 → Layer 1: audio buffers feed recognition
        audio.onBuffer = { [weak self] buffer in
            guard self?.isMuted != true else { return }
            speech.appendBuffer(buffer)
            Task { @MainActor in
                audioDetection?.processBuffer(buffer)
            }
        }

        // Separately forward the AVAudioTime so the STT backend can stamp an
        // accurate t=0 for its analyzer timeline on the first buffer.
        audio.onBufferWithTime = { [weak self] buffer, when in
            guard self?.isMuted != true else { return }
            speech.noteBufferTime(buffer: buffer, when: when)
        }

        // Layer 0: engine restart → restart recognition
        // Triggered by AudioEngineManager (internal engine health-check /
        // interruption ended). For apps using an EXTERNAL AVAudioEngine
        // (usesExternalEngine=true), this callback will NOT fire — the
        // host app must call `notifyAudioStreamRestarted()` directly from
        // its own engine restart path.
        audio.onRestart = { [weak self] in
            Task { @MainActor in
                self?.notifyAudioStreamRestarted()
            }
        }

        // Layer 1 → Layer 2: transcription results feed sentence builder
        speech.onResult = { [weak self] text, isFinal, speakerLabel, audioStartTime, audioEndTime in
            guard let self else { return }
            self.currentSpeakerLabel = speakerLabel
            self.builder.handleResult(text: text, isFinal: isFinal, audioStartTime: audioStartTime, audioEndTime: audioEndTime)
            if isFinal {
                self.onSentenceFinalized?(text)
                self.builder.resetActive()
            }
            self.rebuildSegments()
        }

        // SpeechDetector VAD callback (only available with SpeechAnalyzer backend)
        if let analyzerBackend = speech as? SpeechAnalyzerBackend {
            analyzerBackend.onSpeechDetected = { [weak self] detected in
                self?.speechDetected = detected
            }
            // Propagate contextual strings set before wireUp
            if !contextualStrings.isEmpty {
                analyzerBackend.contextualStrings = contextualStrings
            }
        }

        speech.onTokens = { [weak self] tokens, isFinal in
            self?.onTokens?(tokens, isFinal)
            // Feed token attributor (legacy)
            self?.tokenAttributor.process(tokens: tokens, isFinal: isFinal)
            // Feed new reconstructor
            self?.userSentenceReconstructor.recordTokens(
                tokens, isFinal: isFinal,
                audioStreamStart: self?.audioStreamStartTime
            )
        }

        speech.onFirstBuffer = { [weak self] date in
            // Replace the coarse approximation stamped in start() with the
            // precise moment the analyzer actually consumed its first buffer.
            self?.audioStreamStartTime = date
            self?.tokenAttributor.audioStreamStartTime = date
        }

        speech.onError = { [weak self] error in
            self?.lastError = error.localizedDescription
            self?.builder.finalizeAndReset()
            self?.rebuildSegments()
        }
    }

    /// When true (default), rebuildSegments() post-processes the builder's
    /// output and overrides `isFromUser` using UserSentenceReconstructor's
    /// per-token attribution when the new engine has coverage for the
    /// segment's audio time range. Set to false to opt out and keep the
    /// legacy (LipAudioCorrelator-based) attribution.
    public var useReconstructorForSegmentAttribution: Bool = true
    /// Minimum user-token ratio required (across reconstructor tokens that
    /// overlap a segment's audioTimeRange) for the segment to be marked
    /// `isFromUser = true`. Keep aligned with reconstructor's own
    /// sentence-vote threshold.
    public var reconstructorSegmentUserRatio: Float = 0.4
    /// When true (default), also replaces `SpeechSegment.text` with the
    /// user-only subset produced by UserSentenceReconstructor for the
    /// segment's audio time range. This gives apps rendering
    /// `segment.text` a live "what the user just said to the screen"
    /// stream without any code change. Only takes effect when
    /// `useReconstructorForSegmentAttribution` is also true and the
    /// segment is flagged as user.
    public var useReconstructorForSegmentText: Bool = true

    private func rebuildSegments() {
        let raw = builder.buildSegments()
        guard useReconstructorForSegmentAttribution else {
            segments = raw
            return
        }
        // Convert segment audio-stream-relative times to wall-clock so they
        // match the reconstructor's internal TokenRow.startTime/endTime
        // (which are audioStreamStartTime + audio-offset seconds).
        //
        // Without this conversion, every query misses every token and
        // `attr.hasCoverage` is always false — segments silently retain
        // the builder's legacy LipAudioCorrelator-based isFromUser verdict,
        // even when the reconstructor clearly has the tokens attributed to
        // the user (this was observed in kenefe's side-by-side demo vs
        // visual-talk-ios screenshots: identical state, demo shows ratio=100%,
        // visual-talk-ios shows isFromUser=false).
        let base = audioStreamStartTime?.timeIntervalSince1970 ?? 0

        // SentenceBuilder slices one sentence into multiple SpeechSegments
        // along gaze-span boundaries, and every slice carries the ENTIRE
        // sentence's audioStartTime/audioEndTime (no per-character timing
        // available). If we naively ran userTextInRange per slice, each
        // slice would get the same full user-only text back, duplicating
        // it across the stream.
        //
        // Trick: run the user-only text replacement only once per unique
        // audio-range signature, on the FIRST slice. Subsequent slices of
        // the same sentence get their text zeroed to "". The on-screen
        // result is the same user sentence appearing in the first slice,
        // no duplication.
        var seenRanges: Set<String> = []
        segments = raw.map { seg -> SpeechSegment in
            let wallStart = seg.audioStartTime.map { base + $0 }
            let wallEnd = seg.audioEndTime.map { base + $0 }
            let attr = userSentenceReconstructor.attribution(
                for: wallStart,
                end: wallEnd
            )
            // Only override when the reconstructor actually has coverage for
            // this segment's time range. Otherwise preserve builder's verdict
            // so early segments (before any tokens) don't get false-downgraded.
            guard attr.hasCoverage else { return seg }
            let upgraded = attr.userRatio >= reconstructorSegmentUserRatio

            // Unique signature per (start,end) pair — slices of the same
            // sentence share both.
            let rangeKey: String? = {
                guard let s = wallStart, let e = wallEnd else { return nil }
                return "\(s)_\(e)"
            }()
            let isFirstSliceOfRange: Bool = {
                guard let k = rangeKey else { return true }
                return seenRanges.insert(k).inserted
            }()

            let needsAttrOverride = seg.isFromUser != upgraded
            let preserveBoundary = seg.text == " "
            let shouldReplaceText = useReconstructorForSegmentText && !preserveBoundary
            if !needsAttrOverride && !shouldReplaceText { return seg }

            // Compute the user-only text when requested. userTextInRange
            // filters by isUserWithConfidence (4.9.53), so it naturally
            // returns "" when no token in the range is attributed to the
            // user. Do it only on the FIRST slice of a given audio range
            // so later gaze-span slices don't duplicate the same text.
            //
            // For volatile segments (isFinal=false), pass recentWindowSec
            // so only the most recent 2 seconds of user speech appear.
            // This prevents the "别人说 AAAA BBBB，你说 C，整段 AAAA BBBB C
            // 都出现" problem kenefe reported: when the user starts speaking
            // in the middle of a volatile batch that already contains non-user
            // text, we don't want to display the stale non-user prefix.
            let newText: String
            if shouldReplaceText, isFirstSliceOfRange,
               let userOnly = userSentenceReconstructor.userTextInRange(
                   start: wallStart, end: wallEnd,
                   recentWindowSec: seg.isFinal ? nil : 2.0
               ) {
                newText = userOnly
            } else if shouldReplaceText && !isFirstSliceOfRange {
                // Subsequent slice of the same sentence: empty to avoid
                // duplication across gaze-span slices.
                newText = ""
            } else {
                newText = seg.text
            }

            return SpeechSegment(
                id: seg.id,
                text: newText,
                isToScreen: seg.isToScreen,
                sentenceStartedLookingAtScreen: seg.sentenceStartedLookingAtScreen,
                speakingToAIScore: seg.speakingToAIScore,
                isFromUser: upgraded,
                isFinal: seg.isFinal,
                signals: seg.signals,
                audioStartTime: seg.audioStartTime,
                audioEndTime: seg.audioEndTime
            )
        }
    }
}
#endif
