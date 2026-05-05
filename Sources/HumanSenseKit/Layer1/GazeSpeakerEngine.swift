#if os(iOS)
import SwiftUI
import AVFoundation
import Combine

/// High-level speaker diarization engine that wraps GazeSpeakerAttributor.
/// Manages calibration flow, STT token processing, transcript segments,
/// and debug info. Designed to be the single integration point for apps
/// that need speaker attribution.
@MainActor
@Observable
public class GazeSpeakerEngine: SpeakerAttributionBackend {
    public enum Phase {
        case calibration
        case live
    }

    public struct DebugInfo {
        public var isLookingAtScreen: Bool = false
        public var isHeadForward: Bool = false
        public var speakerMatch: Bool = false
        public var speakerDistance: Float = 1.0
        public var audioLevel: Float = -60.0
        public var userEmbeddingStatus: String = "未标定"
        public var currentJawDelta: Float = 0.0
        public var currentJawVelocity: Float = 0.0
        public var gazeOnScreen: Float = 0.0
        public var headYaw: Float = 0.0
        public var headPitch: Float = 0.0
        public var faceDistance: Float = 0.0
        public init() {}
    }

    public struct TranscriptSegment: Identifiable {
        public let id = UUID()
        public let tokens: [TokenSegment]
        public let isFinal: Bool
        public let timestamp: Date

        public var text: String { tokens.map { $0.text }.joined() }
        public var isUserSpeaker: Bool {
            let userCount = tokens.filter { $0.isUserSpeaker }.count
            return userCount > tokens.count / 2
        }
        public var score: Float {
            tokens.map { $0.score }.reduce(0, +) / Float(tokens.count)
        }
        public var audioTime: Double { tokens.first?.audioTime ?? 0 }

        public init(tokens: [TokenSegment], isFinal: Bool, timestamp: Date) {
            self.tokens = tokens
            self.isFinal = isFinal
            self.timestamp = timestamp
        }
    }

    public struct TokenSegment: Identifiable {
        public let id = UUID()
        public let text: String
        public let isUserSpeaker: Bool
        public let score: Float
        public let audioTime: Double
        public let jawDelta: Float
        public let jawVelocity: Float
        public var gazeOnScreen: Float = 0
        public var headYaw: Float = 0
        public var headPitch: Float = 0
        public var faceDistance: Float = 0
        public var audioLevel: Float = -60.0  // dB RMS at token time

        public init(from speakerToken: SpeakerToken) {
            self.text = speakerToken.text
            self.isUserSpeaker = speakerToken.isUserSpeaker
            self.score = speakerToken.score
            self.audioTime = speakerToken.audioTime
            self.jawDelta = speakerToken.jawDelta
            self.jawVelocity = speakerToken.jawVelocity
            self.gazeOnScreen = speakerToken.gazeOnScreen
            self.headYaw = speakerToken.headYaw
            self.headPitch = speakerToken.headPitch
            self.faceDistance = speakerToken.faceDistance
        }

        public init(text: String, isUserSpeaker: Bool, score: Float, audioTime: Double,
             jawDelta: Float, jawVelocity: Float) {
            self.text = text
            self.isUserSpeaker = isUserSpeaker
            self.score = score
            self.audioTime = audioTime
            self.jawDelta = jawDelta
            self.jawVelocity = jawVelocity
        }
    }

    // MARK: - UI State

    public var phase: Phase = .calibration
    public var transcriptSegments: [TranscriptSegment] = []
    public var currentTokens: [TokenSegment] = []
    public var debugInfo = DebugInfo()

    // Pause detection: fire .userSpeech early when volatile tokens
    // stop arriving for >1s (Apple's isFinal can lag 2-3s on Chinese).
    private var pauseTimer: Timer?
    private var sentenceCounter: Int = 0
    private var lastStreamingAttributedTokens: [SpeakerAttributedToken] = []
    /// Pause-commit threshold in seconds. When no new user tokens arrive
    /// for this duration, `.userSpeech` fires early (before isFinal).
    /// Set to 0 to disable pause-commit (only isFinal triggers userSpeech).
    public var pauseCommitThreshold: TimeInterval = 2.0
    /// Dedup: track last emitted userSpeech text to avoid double-firing
    /// when pause commit fires and then isFinal arrives with same text.
    private var lastEmittedUserSpeechText: String = ""
    public var calibrationProgress: Float = 0.0
    public var isCalibrating = false

    // MARK: - SpeakerAttributionBackend conformance

    /// Unified phase exposed to consumers via the protocol.
    public var speakerPhase: SpeakerEnginePhase {
        if isCalibrating {
            let sentences = calibrationSentences
            let idx = currentCalibrationSentence
            let sentence = idx < sentences.count ? sentences[idx] : nil
            return .calibrating(progress: calibrationProgress, currentSentence: sentence,
                                sentenceIndex: idx, totalSentences: sentences.count)
        }
        switch phase {
        case .calibration: return .unconfigured
        case .live: return .ready
        }
    }

    /// Attributed segments for protocol consumers.
    public var attributedSegments: [SpeakerAttributedSegment] {
        transcriptSegments.map { seg in
            SpeakerAttributedSegment(
                tokens: seg.tokens.map { Self.toAttributedToken($0) },
                isFinal: seg.isFinal,
                timestamp: seg.timestamp
            )
        }
    }

    /// Attributed current tokens for protocol consumers.
    public var attributedCurrentTokens: [SpeakerAttributedToken] {
        currentTokens.map { Self.toAttributedToken($0) }
    }

    /// Debug info for protocol consumers.
    public var speakerDebugInfo: SpeakerDebugInfo {
        SpeakerDebugInfo(
            userEmbeddingStatus: debugInfo.userEmbeddingStatus,
            embeddingCount: attributor?.embeddingCount ?? 0,
            ttsWindowActive: ttsWindowActive,
            lastUserSimilarity: debugInfo.speakerDistance,
            lastJawDelta: debugInfo.currentJawDelta
        )
    }

    private let _eventSubject = PassthroughSubject<SpeakerEvent, Never>()
    public var events: AnyPublisher<SpeakerEvent, Never> { _eventSubject.eraseToAnyPublisher() }

    // TTS window state
    private var ttsWindowActive: Bool = false
    private var ttsWindowEndedAt: Date? = nil
    private let ttsTailGraceSeconds: TimeInterval = 0.2
    private var ttsExpectedText: String? = nil

    // MARK: - Delegated thresholds (proxy to attributor)

    public var speakerThreshold: Float {
        get { attributor?.speakerThreshold ?? 5.75 }
        set { attributor?.speakerThreshold = newValue }
    }
    public var perTokenThreshold: Float {
        get { attributor?.perTokenThreshold ?? 5.75 }
        set { attributor?.perTokenThreshold = newValue }
    }
    public var scoreWeight: Float {
        get { attributor?.scoreWeight ?? 0.5 }
        set { attributor?.scoreWeight = newValue }
    }
    public var jawWeight: Float {
        get { attributor?.jawWeight ?? 1.5 }
        set { attributor?.jawWeight = newValue }
    }
    public var jawVelocityWeight: Float {
        get { attributor?.jawVelocityWeight ?? 2.5 }
        set { attributor?.jawVelocityWeight = newValue }
    }
    public var timeDeltaWeight: Float {
        get { attributor?.timeDeltaWeight ?? 0.5 }
        set { attributor?.timeDeltaWeight = newValue }
    }
    public var contextWeight: Float {
        get { attributor?.contextWeight ?? 0.25 }
        set { attributor?.contextWeight = newValue }
    }
    public var jawMargin: Double {
        get { attributor?.jawMargin ?? 0.1 }
        set { attributor?.jawMargin = newValue }
    }
    public var noJawPenalty: Float {
        get { attributor?.noJawPenalty ?? 0.5 }
        set { attributor?.noJawPenalty = newValue }
    }
    public var enableIncrementalLearning: Bool {
        get { attributor?.enableIncrementalLearning ?? true }
        set { attributor?.enableIncrementalLearning = newValue }
    }
    public var learningThreshold: Float {
        get { attributor?.learningThreshold ?? 4.0 }
        set { attributor?.learningThreshold = newValue }
    }
    public var learningRate: Float {
        get { attributor?.learningRate ?? 0.3 }
        set { attributor?.learningRate = newValue }
    }
    public var learningCount: Int { attributor?.learningCount ?? 0 }
    /// Mirror of attributor.currentCalibrationSentence — kept as stored @Observable
    /// property so SwiftUI re-renders when it advances between calibration steps.
    public var currentCalibrationSentence: Int = 0
    public var calibrationSentences: [String] { attributor?.calibrationSentences ?? [] }

    // MARK: - Callback

    /// Called when user speech is finalized. Apps can use this to trigger AI response.
    public var onUserSpeech: ((_ text: String) -> Void)?

    // MARK: - Private

    private let engine: HumanStateEngine
    private var attributor: GazeSpeakerAttributor?
    private var attributorCancellables = Set<AnyCancellable>()
    private var audioStreamStartTime: Date?

    private let logFileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("speaker_recognition_log.jsonl")
    }()

    // MARK: - Init

    nonisolated public init(engine: HumanStateEngine) {
        self.engine = engine

        Task { @MainActor in
            self.attributor = GazeSpeakerAttributor()
            self.observeAttributor()
            self.setupSTTListener()
            self.setupAudioStream()
            self.syncFromAttributor()
        }
    }

    private func syncFromAttributor() {
        guard let attributor = attributor else { return }
        if attributor.hasEmbedding {
            phase = .live
            debugInfo.userEmbeddingStatus = "✅ 已加载 (\(attributor.embeddingCount) 个样本)"
        }
    }

    /// Observe attributor @Published changes to keep engine state in sync.
    private func observeAttributor() {
        guard let attributor = attributor else { return }

        attributor.$isCalibrating
            .receive(on: DispatchQueue.main)
            .sink { [weak self] calibrating in
                guard let self else { return }
                self.isCalibrating = calibrating
                switch attributor.phase {
                case .calibration: self.phase = .calibration
                case .live: self.phase = .live
                }
                if attributor.hasEmbedding {
                    self.debugInfo.userEmbeddingStatus = "✅ 已标定 (\(attributor.embeddingCount) 个样本)"
                } else if calibrating {
                    self.debugInfo.userEmbeddingStatus = "标定中 (\(attributor.currentCalibrationSentence + 1)/\(attributor.calibrationSentences.count))..."
                } else {
                    self.debugInfo.userEmbeddingStatus = "未标定"
                }
                self._eventSubject.send(.phaseChanged(self.speakerPhase))
            }
            .store(in: &attributorCancellables)

        attributor.$currentCalibrationSentence
            .receive(on: DispatchQueue.main)
            .sink { [weak self] idx in
                guard let self else { return }
                self.currentCalibrationSentence = idx
                if self.isCalibrating, !attributor.hasEmbedding {
                    self.debugInfo.userEmbeddingStatus = "标定中 (\(idx + 1)/\(attributor.calibrationSentences.count))..."
                }
            }
            .store(in: &attributorCancellables)

        attributor.$calibrationProgress
            .receive(on: DispatchQueue.main)
            .sink { [weak self] progress in
                self?.calibrationProgress = progress
            }
            .store(in: &attributorCancellables)
    }

    /// Auto-wire STTManager's audio stream for embedding extraction.
    private func setupAudioStream() {
        engine.sttManager.onAudioSamples = { [weak self] samples in
            self?.processAudioBuffer(samples)
        }
    }

    // MARK: - STT Listener

    private func setupSTTListener() {
        audioStreamStartTime = engine.sttManager.audioStreamStartTime

        let onTokens: ((_ tokens: [SpeechToken], _ isFinal: Bool) -> Void) = { [weak self] tokens, isFinal in
            guard let self else { return }
            guard self.phase == .live, !self.isCalibrating, !tokens.isEmpty else { return }

            Task { @MainActor in
                guard let attributor = self.attributor else { return }
                let speakerTokens = attributor.processTokens(tokens, isFinal: isFinal)
                let newTokens = speakerTokens.map { TokenSegment(from: $0) }

                // Build source-resolved attributed tokens (TTS window logic
                // applied here; legacy TokenSegment.isUserSpeaker stays untouched
                // so old UI code + transcriptSegments grouping keeps working).
                let attributedTokens = newTokens.map { legacy in
                    Self.toAttributedToken(legacy, source: self.resolveSource(for: legacy))
                }

                if isFinal {
                    self.pauseTimer?.invalidate()
                    self.pauseTimer = nil
                    self.lastStreamingAttributedTokens = []
                    self.buildFinalSegments(newTokens)
                    self.currentTokens = []

                    // Two-layer classification (v10d):
                    // 1. Token-level: each token gets user/ambient independently
                    // 2. Sentence-level: overall sentence classification
                    let classifiedTokens = self.classifyTokens(newTokens, isFinal: true)
                    let overriddenTokens = zip(attributedTokens, classifiedTokens).map { (attr, isUser) in
                        SpeakerAttributedToken(text: attr.text, audioTime: attr.audioTime, endTime: attr.endTime,
                                               source: isUser ? .user : .ambient, score: attr.score, metadata: attr.metadata)
                    }

                    let segment = SpeakerAttributedSegment(
                        tokens: overriddenTokens,
                        isFinal: true, timestamp: Date()
                    )
                    let userTokens = overriddenTokens.filter { $0.source == .user }
                    if !userTokens.isEmpty {
                        self._eventSubject.send(.finalSegment(segment))
                    }

                    let userText = userTokens.map(\.text).joined()
                    // Dedup: skip if pause-commit already emitted this (or a close variant).
                    // STT often revises text between pause-commit and final (e.g. "现在放假呢" vs "现放假呢"),
                    // so use containment check instead of exact equality.
                    let isDuplicate: Bool = {
                        guard !userText.isEmpty, !self.lastEmittedUserSpeechText.isEmpty else { return false }
                        let prev = self.lastEmittedUserSpeechText
                        // Either one contains the other, or they share >60% characters
                        if prev.contains(userText) || userText.contains(prev) { return true }
                        let common = Set(prev).intersection(Set(userText)).count
                        let maxLen = max(prev.count, userText.count)
                        return maxLen > 0 && Float(common) / Float(maxLen) > 0.6
                    }()
                    if !userText.isEmpty && !isDuplicate {
                        self.lastEmittedUserSpeechText = userText
                        self.onUserSpeech?(userText)
                        self._eventSubject.send(.userSpeech(text: userText, segment: segment))
                    }
                    // Reset dedup after final — next utterance is a new turn.
                    self.lastEmittedUserSpeechText = ""

                    // Always rotate STT task after isFinal, regardless of
                    // whether userSpeech was emitted. Without this, the next
                    // utterance's tokens get appended to the old task's stream,
                    // mixing "真的吗" residue into "今天天气怎么样".
                    self.engine.sttManager.rotateTask()
                    self.sentenceCounter += 1
                } else {
                    self.buildStreamingTokens(newTokens)

                    // Two-layer classification (v10d) for streaming.
                    let classifiedTokens = self.classifyTokens(self.currentTokens, isFinal: false)
                    let hasUserTokens = classifiedTokens.contains(true)

                    if hasUserTokens {
                        // Emit tokens with per-token classification
                        let classified = zip(self.currentTokens, classifiedTokens).map { (legacy, isUser) in
                            let attr = Self.toAttributedToken(legacy, source: isUser ? .user : .ambient)
                            return SpeakerAttributedToken(text: attr.text, audioTime: attr.audioTime, endTime: attr.endTime,
                                                          source: isUser ? .user : .ambient, score: attr.score, metadata: attr.metadata)
                        }
                        self._eventSubject.send(.streamingTokens(classified))
                    }
                    // If no user tokens, don't emit streaming tokens (silence)

                    // Pause detection: only reset timer if sentence has user tokens.
                    if hasUserTokens && self.pauseCommitThreshold > 0 {
                        let accumulatedAttributed = zip(self.currentTokens, classifiedTokens).map { (legacy, isUser) in
                            let attr = Self.toAttributedToken(legacy, source: isUser ? .user : .ambient)
                            return SpeakerAttributedToken(text: attr.text, audioTime: attr.audioTime, endTime: attr.endTime,
                                                          source: isUser ? .user : .ambient, score: attr.score, metadata: attr.metadata)
                        }
                        self.lastStreamingAttributedTokens = accumulatedAttributed
                        self.pauseTimer?.invalidate()
                        self.pauseTimer = Timer.scheduledTimer(
                            withTimeInterval: self.pauseCommitThreshold, repeats: false
                        ) { [weak self] _ in
                            Task { @MainActor [weak self] in
                                self?.firePauseCommit()
                            }
                        }
                    }
                }
            }
        }
        engine.sttManager.onTokens = onTokens
    }

    /// Pause commit: volatile tokens stopped arriving for >1s.
    /// Emit `.userSpeech` early so the consumer (VM) can start LLM
    /// without waiting for Apple's isFinal (which can lag 2-3s on Chinese).
    /// When isFinal eventually arrives, the duplicate is harmless — VM
    /// will see the same text and can deduplicate or treat as confirmation.
    private func firePauseCommit() {
        pauseTimer = nil
        let tokens = lastStreamingAttributedTokens
        lastStreamingAttributedTokens = []
        guard !tokens.isEmpty else { return }

        // Sandwich repair: if a non-user token sits between two user tokens
        // in a continuous utterance, it's almost certainly the same speaker.
        // Promote isolated non-user tokens to .user when surrounded by .user.
        let repaired = Self.sandwichRepair(tokens)

        let userText = repaired
            .filter { $0.source == .user }
            .map(\.text).joined()
        guard !userText.isEmpty else { return }

        // Fragment-drop guard: if user tokens are still < 50% after repair,
        // the whole segment is likely ambient/tts — drop it.
        let totalText = repaired.map(\.text).joined()
        let userRatio: Double = totalText.isEmpty ? 0 :
            Double(userText.count) / Double(totalText.count)
        if !totalText.isEmpty && userRatio < 0.5 {
            NSLog("[GazeSpeakerEngine] fragment drop (pause): userRatio=%.2f userText='%@' totalText='%@'",
                  userRatio, userText, totalText)
            return
        }

        lastEmittedUserSpeechText = userText
        let segment = SpeakerAttributedSegment(
            tokens: repaired, isFinal: false, timestamp: Date()
        )
        onUserSpeech?(userText)
        _eventSubject.send(.userSpeech(text: userText, segment: segment))
    }

    /// Three-source classification for an incoming token.
    ///
    /// Outside TTS window: user vs ambient (follows attributor's isUserSpeaker).
    /// Inside TTS window: demote to .tts unless the attributor's combined score
    /// is *well above* threshold (strong barge-in) OR text overlaps the TTS
    /// output. AEC residue + speaker-embedding mismatch will naturally sit
    /// near the threshold boundary, so a margin of 1.3x keeps it out of .user.
    private func resolveSource(for token: TokenSegment) -> SpeakerSource {
        let now = Date()
        let inWindow = ttsWindowActive ||
            (ttsWindowEndedAt.map { now.timeIntervalSince($0) < ttsTailGraceSeconds } ?? false)

        guard inWindow else {
            return token.isUserSpeaker ? .user : .ambient
        }

        // TTS window active — tighten the user-acceptance criterion.
        let baseThreshold = attributor?.speakerThreshold ?? 4.75
        let strongCutoff = baseThreshold * 1.3   // score must be well above normal pass
        if token.isUserSpeaker && token.score > strongCutoff {
            return .user   // strong barge-in
        }

        // If expectedText overlaps the transcribed token text, it's almost
        // certainly AI echo, regardless of what the attributor thinks.
        if let expected = ttsExpectedText, !expected.isEmpty,
           Self.tokenEchoesExpected(token.text, expected: expected) {
            return .tts
        }

        // Default during TTS window: attributor said user but not strongly
        // enough — treat as AEC leak (.tts). Non-user stays ambient; this is
        // rare because AEC + silence usually keep ambient tokens from reaching
        // here, but if a second real person talks during TTS we still label
        // them .ambient (not .tts) so downstream can decide.
        if token.isUserSpeaker {
            return .tts   // weak user match during TTS ≈ echo
        }
        return .ambient
    }

    /// Sandwich repair: promote isolated non-user tokens to .user when they
    /// sit between two .user tokens in a continuous utterance. Handles the
    /// common case where attributor misjudges a single syllable mid-sentence
    /// (e.g. "怎么样啊？" → all ✗ except nothing, but surrounded by user tokens).
    ///
    /// Also handles the "majority user" case: if ≥50% of tokens are already
    /// .user, promote ALL non-user tokens (the speaker didn't change mid-sentence).
    private static func sandwichRepair(_ tokens: [SpeakerAttributedToken]) -> [SpeakerAttributedToken] {
        guard tokens.count >= 2 else { return tokens }
        let userCount = tokens.filter { $0.source == .user }.count
        // Majority rule: if ≥50% are user, promote everything
        if userCount * 2 >= tokens.count {
            return tokens.map { t in
                t.source == .user ? t : SpeakerAttributedToken(
                    id: t.id, text: t.text, audioTime: t.audioTime,
                    endTime: t.endTime, source: .user, score: t.score,
                    metadata: t.metadata)
            }
        }
        // Sandwich: promote non-user tokens between two user tokens
        var result = tokens
        for i in 1..<(tokens.count - 1) {
            if tokens[i].source != .user,
               tokens[i-1].source == .user,
               tokens[i+1].source == .user {
                result[i] = SpeakerAttributedToken(
                    id: tokens[i].id, text: tokens[i].text,
                    audioTime: tokens[i].audioTime, endTime: tokens[i].endTime,
                    source: .user, score: tokens[i].score,
                    metadata: tokens[i].metadata)
            }
        }
        return result
    }

    /// Very cheap substring check: does the transcribed token appear inside
    /// the expected TTS text? Case-insensitive, trims whitespace. Short tokens
    /// (< 2 chars) skip this heuristic to avoid false positives on common
    /// Chinese syllables like "的" / "了".
    private static func tokenEchoesExpected(_ tokenText: String, expected: String) -> Bool {
        let t = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard t.count >= 2 else { return false }
        return expected.localizedCaseInsensitiveContains(t)
    }

    private func buildFinalSegments(_ newTokens: [TokenSegment]) {
        // isFinal tokens represent the complete, corrected utterance.
        // Store ALL tokens so debug UI has the full picture.
        // User-only filtering happens at event emission.
        guard !newTokens.isEmpty else { return }

        for token in newTokens {
            logTokenRecognition(token: token, isFinal: true)
        }

        transcriptSegments.append(TranscriptSegment(
            tokens: newTokens, isFinal: true, timestamp: Date()
        ))
        if transcriptSegments.count > 20 {
            transcriptSegments.removeFirst(transcriptSegments.count - 20)
        }
    }

    private func buildStreamingTokens(_ newTokens: [TokenSegment]) {
        // Apple STT (timeIndexedProgressiveTranscription) returns the FULL
        // utterance on every callback, not incremental deltas. Each call
        // replaces the previous tokens entirely — corrections to earlier
        // words arrive as updated tokens with the same or shifted audioTime.
        //
        // Store ALL tokens here (including non-user) so debug UI and
        // attributor have the full picture. User-only filtering happens
        // at event emission (.streamingTokens, .finalSegment, .userSpeech).
        currentTokens = newTokens
        #if DEBUG
        for token in newTokens {
            logTokenRecognition(token: token, isFinal: false)
        }
        #endif
    }

    // MARK: - Audio Processing

    public func processAudioBuffer(_ samples: [Float]) {
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        debugInfo.audioLevel = 20 * log10(max(rms, 1e-10))

        let face = engine.humanState.face

        if !isCalibrating, phase == .live,
           let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            attributor?.recordSensorData(face: face, audioStreamElapsed: elapsed)
        }

        let elapsed: Double? = engine.sttManager.audioStreamStartTime.map {
            Date().timeIntervalSince($0)
        }
        attributor?.processAudioForEmbedding(samples, face: face, audioStreamElapsed: elapsed)

        syncDebugInfo(face: face)
    }

    private func syncDebugInfo(face: FaceState) {
        debugInfo.isLookingAtScreen = face.isLookingAtScreen
        debugInfo.isHeadForward = face.headOrientation.isFacingForward
        debugInfo.gazeOnScreen = face.isLookingAtScreen ? 1.0 : 0.0
        debugInfo.headYaw = face.headYaw
        debugInfo.headPitch = face.headPitch
        debugInfo.faceDistance = face.distanceFromCamera
        debugInfo.speakerMatch = attributor?.speakerMatch ?? false
        debugInfo.speakerDistance = attributor?.speakerDistance ?? 1.0

        isCalibrating = attributor?.isCalibrating ?? false
        calibrationProgress = attributor?.calibrationProgress ?? 0

        switch attributor?.phase {
        case .calibration: phase = .calibration
        case .live: phase = .live
        case .none: break
        }

        if attributor?.hasEmbedding == true {
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(attributor?.embeddingCount ?? 0) 个样本)"
        } else if isCalibrating {
            debugInfo.userEmbeddingStatus = "标定中 (\((attributor?.currentCalibrationSentence ?? 0) + 1)/\(attributor?.calibrationSentences.count ?? 0))..."
        }

        debugInfo.currentJawDelta = face.jawOpen
        debugInfo.currentJawVelocity = 0
    }

    // MARK: - Calibration

    public func startCalibration() {
        attributor?.startCalibration()
        isCalibrating = true
        debugInfo.userEmbeddingStatus = "标定中 (1/\(calibrationSentences.count))..."
    }

    public func startAdditionalCalibration() {
        attributor?.startAdditionalCalibration()
        isCalibrating = true
        debugInfo.userEmbeddingStatus = "追加标定中..."
    }

    public func stopAdditionalCalibration() {
        attributor?.stopAdditionalCalibration()
        isCalibrating = false
        if attributor?.hasEmbedding == true {
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(attributor?.embeddingCount ?? 0) 个样本)"
        }
    }

    public func resetToInitialEmbedding() {
        attributor?.resetToInitialEmbedding()
    }

    public func reset() {
        attributor?.reset()
    }

    public func deleteEmbedding() {
        attributor?.deleteEmbedding()
        phase = .calibration
        debugInfo.userEmbeddingStatus = "未标定"
    }

    public func clearTranscript() {
        transcriptSegments = []
        currentTokens = []
        attributor?.reset()
    }

    // MARK: - SpeakerAttributionBackend Protocol

    /// Provide `phase` conformance by mapping to `SpeakerEnginePhase`.
    /// Note: protocol requires `var phase: SpeakerEnginePhase` but we already have
    /// `var phase: Phase`. We satisfy the protocol via explicit witness methods below.

    public func cancelCalibration() {
        attributor?.stopAdditionalCalibration()
        isCalibrating = false
        _eventSubject.send(.phaseChanged(speakerPhase))
    }

    public func resetCalibration() {
        deleteEmbedding()
        _eventSubject.send(.phaseChanged(speakerPhase))
    }

    public func markTTSStart(expectedText: String?) {
        ttsWindowActive = true
        ttsExpectedText = expectedText
        ttsWindowEndedAt = nil
    }

    public func markTTSEnd() {
        ttsWindowActive = false
        ttsWindowEndedAt = Date()
    }

    /// Convert a legacy TokenSegment to SpeakerAttributedToken with an
    /// explicit source (three-way: .user / .ambient / .tts) resolved by
    /// `resolveSource`. Keep this overload so callers that don't care about
    /// TTS window (e.g. `attributedSegments` getter for display) still work.
    private static func toAttributedToken(_ token: TokenSegment,
                                          source: SpeakerSource? = nil) -> SpeakerAttributedToken {
        let resolved = source ?? (token.isUserSpeaker ? .user : .ambient)
        let metadata = SpeakerTokenMetadata(
            jawDelta: token.jawDelta,
            jawVelocity: token.jawVelocity,
            gazeOnScreen: token.gazeOnScreen,
            headYaw: token.headYaw,
            headPitch: token.headPitch,
            faceDistance: token.faceDistance
        )
        return SpeakerAttributedToken(
            text: token.text,
            audioTime: token.audioTime,
            endTime: token.audioTime,
            source: resolved,
            score: token.score,
            metadata: metadata
        )
    }

    // MARK: - Adaptive Sentence Classification (v10d)

    /// Two-layer adaptive classifier with jaw variance and gaze refinements:
    /// 1. Scene detection: conversation vs media (jaw p25 < 0.05)
    /// 2a. Conversation: decision tree + jaw_std penalty + gaze gate on strong accept
    /// 2b. Media: token-level jaw spike detection
    ///
    /// Autoresearch result (4 test files, 40 sentences):
    /// - Sentence-level (stream-only): Acc=92.5%, Prec=83.3%, Rec=100%, F1=90.9%
    /// - Per-phase: Acc=91.0%, Prec=87.1%, Rec=90.0%, F1=88.5%
    private static let jawVelocityCap: Float = 5.0
    private static let mediaSceneP25Threshold: Float = 0.05

    /// Running jaw values for scene detection (last ~200 tokens)
    private var recentJawValues: [Float] = []
    private static let jawHistorySize = 200

    private var isMediaScene: Bool {
        guard recentJawValues.count >= 20 else { return false }
        let sorted = recentJawValues.sorted()
        let p25 = sorted[sorted.count / 4]
        return p25 < Self.mediaSceneP25Threshold
    }

    private func updateJawHistory(_ tokens: [TokenSegment]) {
        for t in tokens {
            recentJawValues.append(t.jawDelta)
        }
        if recentJawValues.count > Self.jawHistorySize {
            recentJawValues.removeFirst(recentJawValues.count - Self.jawHistorySize)
        }
    }

    func classifySentenceAsUser(_ tokens: [TokenSegment], isFinal: Bool) -> Bool {
        let tokenResults = classifyTokens(tokens, isFinal: isFinal)
        return tokenResults.contains(true)
    }

    /// Media scene: kept for reference. Logic moved to classifyTokens + classifyTokenInMediaScene.

    /// Conversation scene: pitch-aware decision tree (v3b).
    ///
    /// Key insight from autoresearch on 51 sentences (4 user, 47 ambient):
    /// - pitch (head tilt) is the strongest separator: user < 0.44, ambient > 0.49
    /// - gaze remains important: user > 0.41, most ambient < 0.30
    /// - jaw alone is insufficient: user jaw 0.20-0.30, ambient jaw 0.21-0.40 (overlap!)
    ///
    /// Performance: Precision=100% Recall=100% F1=1.000 FPR=0.0%
    /// (vs v2: Precision=14% Recall=50% F1=0.222 FPR=25.5%)
    private func classifyConversationScene(_ tokens: [TokenSegment], isFinal: Bool) -> Bool {
        // Session gate for FINAL only: require sessionIsUserSpeaking before
        // triggering LLM. Streaming tokens are shown more permissively so
        // the user gets immediate visual feedback while session bootstraps.
        if isFinal && !engine.sessionIsUserSpeaking {
            return false
        }

        let n = Float(tokens.count)

        let jawMean = tokens.reduce(Float(0)) { $0 + $1.jawDelta } / n
        let gazeMean = tokens.reduce(Float(0)) { $0 + $1.gazeOnScreen } / n
        let yawMean = tokens.reduce(Float(0)) { $0 + abs($1.headYaw) } / n
        let distMean = tokens.reduce(Float(0)) { $0 + $1.faceDistance } / n
        let pitchMean = tokens.reduce(Float(0)) { $0 + $1.headPitch } / n

        // Jaw variance: high std means fluctuating jaw (likely background noise)
        let jawVariance = tokens.reduce(Float(0)) { $0 + ($1.jawDelta - jawMean) * ($1.jawDelta - jawMean) } / n
        let jawStd = sqrtf(jawVariance)

        // === v5 Multi-Path Classifier ===
        // Optimized on 94 real-world sentences: F1=0.800, Precision=94.7%, Recall=69.2%
        // Two paths: looking at screen (lenient) vs not looking (strict)

        // Hard reject: no jaw activity at all
        if jawMean < 0.15 { return false }

        // Hard reject: turned away
        if yawMean > 0.40 { return false }

        // Hard reject: not looking at screen at all (strongest anti-FP signal)
        // User who is NOT talking to device has gaze < 0.20 typically
        if gazeMean < 0.20 { return false }

        // Path A: Looking at screen (gaze >= 0.35)
        if gazeMean >= 0.35 {
            // Accept if jaw is clearly active
            if jawMean >= 0.20 {
                if jawStd > 0.15 { return false }  // noisy jaw = background
                return true
            }
            // Lower jaw but looking → accept if pitch indicates close/engaged
            if pitchMean < 0.55 {
                return true
            }
            return false
        }

        // Path B: Not looking at screen (gaze 0.20-0.35)
        // Need stronger evidence: jaw + close distance + low yaw
        if jawMean >= 0.30 && distMean < 0.55 && yawMean < 0.25 {
            return true
        }

        return false
    }

    // MARK: - Two-Layer Classification

    /// Classify each token as user/non-user, then apply sentence-level logic.
    /// Returns an array of Bool (one per token) indicating user speech.
    ///
    /// - Media scene: token-level jaw spike detection (each token independent)
    /// - Conversation scene: sentence-level decision tree (all tokens same label)
    func classifyTokens(_ tokens: [TokenSegment], isFinal: Bool) -> [Bool] {
        guard !tokens.isEmpty else { return [] }

        // Update jaw history for scene detection
        updateJawHistory(tokens)

        if isMediaScene {
            // Token-level: classify each token independently by jaw spike
            let tokenResults = tokens.map { classifyTokenInMediaScene($0) }

            // Sentence-level gate: require minimum active ratio
            let activeCount = tokenResults.filter { $0 }.count
            let activeRatio = Float(activeCount) / Float(tokens.count)
            let activeToks = tokens.enumerated().filter { tokenResults[$0.offset] }
            let activeJawV: Float = activeToks.isEmpty ? 0 :
                activeToks.reduce(Float(0)) { $0 + min($1.element.jawVelocity, Self.jawVelocityCap) } / Float(activeToks.count)

            // If too few active tokens or low jawV, reject entire sentence
            if activeRatio < 0.08 || activeJawV < 1.0 {
                return Array(repeating: false, count: tokens.count)
            }
            return tokenResults
        } else {
            // Conversation scene: v5 multi-path sentence-level classification.
            // All tokens get the same label. Per-token split is unreliable in
            // conversation mode because attributor's per-char threshold is too
            // permissive for ambient speech with slight jaw movement.
            let sentenceIsUser = classifyConversationScene(tokens, isFinal: isFinal)
            return Array(repeating: sentenceIsUser, count: tokens.count)
        }
    }

    /// Token-level classification for media scene.
    /// User tokens have jaw >= 0.2 AND short text (background video accumulates long text).
    private func classifyTokenInMediaScene(_ token: TokenSegment) -> Bool {
        return token.jawDelta >= 0.2 && token.text.count <= 15
    }

    // MARK: - Confidence Score

    /// Compute a [0, 1] confidence that the sentence is user speech.
    /// Useful for the app to decide response urgency:
    /// - > 0.8: immediate response
    /// - 0.5-0.8: respond but wait for final confirmation
    /// - < 0.5: likely not user speech
    func sentenceConfidence(_ tokens: [TokenSegment]) -> Float {
        guard !tokens.isEmpty else { return 0 }
        let n = Float(tokens.count)

        let jawMean = tokens.reduce(Float(0)) { $0 + $1.jawDelta } / n
        let jawVMean = tokens.reduce(Float(0)) { $0 + min($1.jawVelocity, Self.jawVelocityCap) } / n
        let gazeMean = tokens.reduce(Float(0)) { $0 + $1.gazeOnScreen } / n
        let yawMean = tokens.reduce(Float(0)) { $0 + abs($1.headYaw) } / n
        let scoreMean = tokens.reduce(Float(0)) { $0 + $1.score } / n
        let jawVariance = tokens.reduce(Float(0)) { $0 + ($1.jawDelta - jawMean) * ($1.jawDelta - jawMean) } / n
        let jawStd = sqrtf(jawVariance)

        var conf: Float = 0
        conf += min(jawMean / 0.4, 1.0) * 0.30       // jaw contribution
        conf += min(gazeMean / 0.5, 1.0) * 0.25       // gaze contribution
        conf += max(0, 1 - yawMean / 0.5) * 0.15      // yaw contribution (lower = better)
        conf += min(jawVMean / 3.0, 1.0) * 0.15       // jawV contribution
        conf += max(0, 1 - jawStd / 0.15) * 0.10      // stability contribution
        conf += min(scoreMean / 0.8, 1.0) * 0.05      // speaker score contribution
        return min(conf, 1.0)
    }

    // MARK: - Logging

    private func logTokenRecognition(token: TokenSegment, isFinal: Bool) {
        // Query lip-audio correlation for this token's time window
        let correlator = engine.lipAudioCorrelator
        let now = ProcessInfo.processInfo.systemUptime
        // Use a small window around current time (token doesn't have exact uptime)
        let lipAudioCorr = correlator.correlation  // current instantaneous correlation

        let logEntry: [String: Any] = [
            "timestamp": Date().timeIntervalSince1970,
            "sentenceId": sentenceCounter,
            "phase": isFinal ? "final" : "streaming",
            "text": token.text,
            "audioTime": token.audioTime,
            "score": token.score,
            "jawDelta": token.jawDelta,
            "jawVelocity": token.jawVelocity,
            "isUserSpeaker": token.isUserSpeaker,
            "isFinal": isFinal,
            "gazeOnScreen": token.gazeOnScreen,
            "headYaw": token.headYaw,
            "headPitch": token.headPitch,
            "faceDistance": token.faceDistance,
            "audioLevel": debugInfo.audioLevel,
            "lipAudioCorr": lipAudioCorr,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: logEntry),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                if let data = (jsonString + "\n").data(using: .utf8) {
                    fileHandle.write(data)
                }
                try? fileHandle.close()
            } else {
                try? (jsonString + "\n").write(to: logFileURL, atomically: true, encoding: .utf8)
            }
        }
    }

    public func getLogFileURL() -> URL { logFileURL }

    public func clearLog() {
            try? FileManager.default.removeItem(at: logFileURL)
    }
}
#endif
