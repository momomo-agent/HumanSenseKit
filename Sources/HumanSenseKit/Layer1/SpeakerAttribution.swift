#if os(iOS)
import Foundation
import Combine

// MARK: - Types

public enum SpeakerSource: String, Sendable, Codable {
    case user       // 本机用户（embedding/jaw/gaze 综合判定）
    case ambient    // 环境人声（不是用户，不是 TTS）
    case tts        // AI 自己的声音回漏（TTS 窗口内）
    case unknown    // 标定未完成 / 数据不足
}

public struct SpeakerAttributedToken: Identifiable, Sendable {
    public let id: UUID
    public let text: String
    public let audioTime: Double
    public let endTime: Double
    public let source: SpeakerSource
    public let score: Float
    public let metadata: SpeakerTokenMetadata

    public init(id: UUID = UUID(), text: String, audioTime: Double, endTime: Double,
                source: SpeakerSource, score: Float, metadata: SpeakerTokenMetadata = .init()) {
        self.id = id
        self.text = text
        self.audioTime = audioTime
        self.endTime = endTime
        self.source = source
        self.score = score
        self.metadata = metadata
    }
}

public struct SpeakerTokenMetadata: Sendable {
    public var jawDelta: Float = 0
    public var jawVelocity: Float = 0
    public var gazeOnScreen: Float = 0
    public var headYaw: Float = 0
    public var headPitch: Float = 0
    public var faceDistance: Float = 0
    public var speakerSimilarity: Float = 0
    public init() {}
    public init(jawDelta: Float, jawVelocity: Float, gazeOnScreen: Float, headYaw: Float, headPitch: Float, faceDistance: Float, speakerSimilarity: Float = 0) {
        self.jawDelta = jawDelta
        self.jawVelocity = jawVelocity
        self.gazeOnScreen = gazeOnScreen
        self.headYaw = headYaw
        self.headPitch = headPitch
        self.faceDistance = faceDistance
        self.speakerSimilarity = speakerSimilarity
    }
}

public struct SpeakerAttributedSegment: Identifiable, Sendable {
    public let id: UUID
    public let tokens: [SpeakerAttributedToken]
    public let isFinal: Bool
    public let timestamp: Date

    public var text: String { tokens.map(\.text).joined() }
    public var userText: String { tokens.filter { $0.source == .user }.map(\.text).joined() }
    public var hasUserSpeech: Bool { tokens.contains { $0.source == .user } }

    public init(id: UUID = UUID(), tokens: [SpeakerAttributedToken], isFinal: Bool, timestamp: Date) {
        self.id = id
        self.tokens = tokens
        self.isFinal = isFinal
        self.timestamp = timestamp
    }
}

public enum SpeakerEnginePhase: Sendable, Equatable {
    case unconfigured
    case calibrating(progress: Float, currentSentence: String?, sentenceIndex: Int, totalSentences: Int)
    case ready

    public static func == (lhs: SpeakerEnginePhase, rhs: SpeakerEnginePhase) -> Bool {
        switch (lhs, rhs) {
        case (.unconfigured, .unconfigured): return true
        case (.ready, .ready): return true
        case (.calibrating(let lp, _, let li, let lt), .calibrating(let rp, _, let ri, let rt)):
            return lp == rp && li == ri && lt == rt
        default: return false
        }
    }
}

public enum SpeakerEvent: Sendable {
    case phaseChanged(SpeakerEnginePhase)
    case streamingTokens([SpeakerAttributedToken])
    case finalSegment(SpeakerAttributedSegment)
    case userSpeech(text: String, segment: SpeakerAttributedSegment)
}

// MARK: - Protocol

@MainActor
public protocol SpeakerAttributionBackend: AnyObject {
    // === State (Observable) ===
    var speakerPhase: SpeakerEnginePhase { get }
    var attributedSegments: [SpeakerAttributedSegment] { get }
    var attributedCurrentTokens: [SpeakerAttributedToken] { get }
    var speakerDebugInfo: SpeakerDebugInfo { get }

    // === Event stream ===
    var events: AnyPublisher<SpeakerEvent, Never> { get }

    // === Calibration ===
    var calibrationSentences: [String] { get }
    func startCalibration()
    func cancelCalibration()
    func resetCalibration()

    // === TTS window ===
    func markTTSStart(expectedText: String?)
    func markTTSEnd()

    // === Control ===
    func reset()
}

public struct SpeakerDebugInfo: Sendable {
    public var userEmbeddingStatus: String = "未标定"
    public var embeddingCount: Int = 0
    public var ttsWindowActive: Bool = false
    public var lastUserSimilarity: Float = 0
    public var lastJawDelta: Float = 0
    public init() {}
    public init(userEmbeddingStatus: String, embeddingCount: Int, ttsWindowActive: Bool, lastUserSimilarity: Float, lastJawDelta: Float) {
        self.userEmbeddingStatus = userEmbeddingStatus
        self.embeddingCount = embeddingCount
        self.ttsWindowActive = ttsWindowActive
        self.lastUserSimilarity = lastUserSimilarity
        self.lastJawDelta = lastJawDelta
    }
}

// MARK: - Legacy Compatibility

/// Typealias for backward compatibility during migration.
public typealias TokenSegment = GazeSpeakerEngine.TokenSegment
#endif
