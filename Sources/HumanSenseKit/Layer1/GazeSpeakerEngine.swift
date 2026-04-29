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
public class GazeSpeakerEngine {
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
    public var calibrationProgress: Float = 0.0
    public var isCalibrating = false

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

                if isFinal {
                    self.buildFinalSegments(newTokens)
                    self.currentTokens = []

                    // Notify callback with user speech
                    let userText = newTokens.filter { $0.isUserSpeaker }.map { $0.text }.joined()
                    if !userText.isEmpty {
                        self.onUserSpeech?(userText)
                    }
                } else {
                    self.buildStreamingTokens(newTokens)
                }
            }
        }
        engine.sttManager.onTokens = onTokens
    }

    private func buildFinalSegments(_ newTokens: [TokenSegment]) {
        var currentGroup: [TokenSegment] = []
        var currentIsUser: Bool? = nil

        if let lastSegment = transcriptSegments.last {
            let timeSinceLastSegment = Date().timeIntervalSince(lastSegment.timestamp)
            if timeSinceLastSegment < 1.0 {
                transcriptSegments.removeLast()
                currentGroup = lastSegment.tokens
                currentIsUser = lastSegment.tokens.last?.isUserSpeaker
            }
        }

        for token in newTokens {
            logTokenRecognition(token: token, isFinal: true)

            if currentIsUser == nil {
                currentIsUser = token.isUserSpeaker
                currentGroup.append(token)
            } else if currentIsUser == token.isUserSpeaker {
                currentGroup.append(token)
            } else {
                if !currentGroup.isEmpty {
                    transcriptSegments.append(TranscriptSegment(
                        tokens: currentGroup, isFinal: true, timestamp: Date()
                    ))
                    if transcriptSegments.count > 20 {
                        transcriptSegments.removeFirst(transcriptSegments.count - 20)
                    }
                }
                currentGroup = [token]
                currentIsUser = token.isUserSpeaker
            }
        }

        if !currentGroup.isEmpty {
            transcriptSegments.append(TranscriptSegment(
                tokens: currentGroup, isFinal: true, timestamp: Date()
            ))
            if transcriptSegments.count > 20 {
                transcriptSegments.removeFirst(transcriptSegments.count - 20)
            }
        }
    }

    private func buildStreamingTokens(_ newTokens: [TokenSegment]) {
        var groupedTokens: [TokenSegment] = []
        var currentGroup: [TokenSegment] = []
        var currentIsUser: Bool? = nil

        for token in newTokens {
            if currentIsUser == nil {
                currentIsUser = token.isUserSpeaker
                currentGroup.append(token)
            } else if currentIsUser == token.isUserSpeaker {
                currentGroup.append(token)
            } else {
                groupedTokens.append(contentsOf: currentGroup)
                currentGroup = [token]
                currentIsUser = token.isUserSpeaker
            }
        }
        groupedTokens.append(contentsOf: currentGroup)

        // Preserve prefix when iOS STT corrects earlier tokens
        if !currentTokens.isEmpty,
           let newFirst = groupedTokens.first,
           let oldFirst = currentTokens.first,
           newFirst.audioTime > oldFirst.audioTime {
            let prefix = currentTokens.filter { $0.audioTime < newFirst.audioTime }
            currentTokens = prefix + groupedTokens
        } else {
            currentTokens = groupedTokens
        }
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

    // MARK: - Logging

    private func logTokenRecognition(token: TokenSegment, isFinal: Bool) {
        let logEntry: [String: Any] = [
            "timestamp": Date().timeIntervalSince1970,
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
