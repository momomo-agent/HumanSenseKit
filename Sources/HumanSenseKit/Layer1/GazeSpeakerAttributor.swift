#if os(iOS)
import Foundation
import Accelerate

/// Output token from speaker diarization with all sensor features.
public struct SpeakerToken {
    public let text: String
    public let isUserSpeaker: Bool
    public let score: Float
    public let audioTime: Double
    public let jawDelta: Float
    public let jawVelocity: Float
    public var gazeOnScreen: Float
    public var headYaw: Float
    public var headPitch: Float
    public var faceDistance: Float

    public init(
        text: String, isUserSpeaker: Bool, score: Float, audioTime: Double,
        jawDelta: Float, jawVelocity: Float,
        gazeOnScreen: Float = 0, headYaw: Float = 0,
        headPitch: Float = 0, faceDistance: Float = 0
    ) {
        self.text = text
        self.isUserSpeaker = isUserSpeaker
        self.score = score
        self.audioTime = audioTime
        self.jawDelta = jawDelta
        self.jawVelocity = jawVelocity
        self.gazeOnScreen = gazeOnScreen
        self.headYaw = headYaw
        self.headPitch = headPitch
        self.faceDistance = faceDistance
    }
}

/// Gaze-gated speaker diarization engine.
/// Uses face tracking (jaw, gaze, head pose) + speaker embeddings to attribute
/// speech tokens to user vs. non-user speakers.
///
/// Usage:
/// 1. Call `recordSensorData` on every face tracking frame
/// 2. Call `processAudioForEmbedding` on every audio buffer
/// 3. Call `processTokens` when STT produces tokens
@MainActor
public class GazeSpeakerAttributor: ObservableObject {

    public enum Phase {
        case calibration
        case live
    }

    // MARK: - Published State

    @Published public var phase: Phase = .calibration
    @Published public var calibrationProgress: Float = 0.0
    @Published public var isCalibrating = false
    @Published public var currentCalibrationSentence: Int = 0

    // MARK: - Public Thresholds (tunable)

    public var speakerThreshold: Float = 4.5
    public var perTokenThreshold: Float = -1.25
    public var scoreWeight: Float = 0.5
    public var jawWeight: Float = 1.5
    public var jawVelocityWeight: Float = 2.5
    public var timeDeltaWeight: Float = 0.5
    public var contextWeight: Float = 0.25
    public var jawMargin: Double = 0.1
    public var noJawPenalty: Float = 0.5
    public var enableIncrementalLearning: Bool = true
    public var learningThreshold: Float = 4.0
    public var learningRate: Float = 0.3

    // MARK: - Public Read-Only

    public private(set) var learningCount: Int = 0
    public private(set) var speakerDistance: Float = 1.0
    public private(set) var speakerMatch: Bool = false

    public var hasEmbedding: Bool { !userEmbeddings.isEmpty }
    public var embeddingCount: Int { userEmbeddings.count }

    public let calibrationSentences = [
        "今天天气真不错",
        "我喜欢听音乐",
        "明天一起去吃饭",
        "这个想法很有趣",
        "请帮我打开窗户"
    ]

    // MARK: - Private State

    private var speakerHistory: [(timestamp: Double, distance: Float)] = []
    private var jawHistory: [(timestamp: Double, jawOpen: Float)] = []
    private var gazeHistory: [(timestamp: Double, onScreen: Bool, yaw: Float, pitch: Float, distance: Float)] = []
    private var tokenColorMap: [String: (isUser: Bool, score: Float)] = [:]
    private var audioBufferQueue: [[Float]] = []
    private let bufferWindowSize: Int = 10
    private var lastTokenAudioTime: Double = 0

    private var userEmbeddings: [[Float]] = [] {
        didSet {
            if !userEmbeddings.isEmpty {
                saveEmbeddings(userEmbeddings)
            }
        }
    }
    private var calibrationAudioBuffers: [[Float]] = []
    private var calibrationEmbeddings: [[Float]] = []
    private let calibrationDuration: TimeInterval = 3.0
    private var calibrationStartTime: Date?
    private var initialEmbedding: [Float]?
    private var lastLearningTime: Date?
    nonisolated(unsafe) private var embeddingExtractor: SimpleSpeakerEmbeddingExtractor?

    private let embeddingFileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("user_speaker_embedding.json")
    }()

    // MARK: - Init

    public init() {
        do {
            self.embeddingExtractor = try SimpleSpeakerEmbeddingExtractor()
        } catch {
            NSLog("[GazeSpeakerAttributor] Failed to load embedding extractor: %@", error.localizedDescription)
            DiarizationDebugLog.write("INIT extractor FAILED: \(error.localizedDescription)")
        }
        loadSavedEmbedding()
    }

    // MARK: - Core Public API

    /// Process speech tokens through diarization. Returns attributed SpeakerTokens.
    public func processTokens(_ tokens: [SpeechToken], isFinal: Bool) -> [SpeakerToken] {
        var allTokens: [SpeakerToken] = []
        for token in tokens {
            let segments = splitTokenBySpeakerChange(token, isFinal: isFinal)
            allTokens.append(contentsOf: segments)
        }

        if isFinal && allTokens.count >= 2 {
            allTokens = smoothSpeakerPredictions(allTokens)
            if enableIncrementalLearning {
                tryIncrementalLearningFromTokens(allTokens)
            }
        }

        return allTokens
    }

    /// Record face sensor data. Call on every face tracking frame.
    public func recordSensorData(face: FaceState, audioStreamElapsed: Double) {
        jawHistory.append((timestamp: audioStreamElapsed, jawOpen: face.jawOpen))
        gazeHistory.append((
            timestamp: audioStreamElapsed,
            onScreen: face.isLookingAtScreen,
            yaw: face.headYaw,
            pitch: face.headPitch,
            distance: face.distanceFromCamera
        ))
        let cutoff = audioStreamElapsed - 10.0
        jawHistory.removeAll { $0.timestamp < cutoff }
        gazeHistory.removeAll { $0.timestamp < cutoff }
    }

    /// Process audio buffer for embedding extraction and speaker comparison.
    public func processAudioForEmbedding(_ samples: [Float], face: FaceState, audioStreamElapsed: Double?) {
        if isCalibrating {
            processCalibrationAudio(samples)
            return
        }
        audioBufferQueue.append(samples)
        if audioBufferQueue.count > bufferWindowSize {
            audioBufferQueue.removeFirst()
        }

        if phase == .live {
            let recentSamples = audioBufferQueue.flatMap { $0 }
            processLiveAudio(recentSamples, face: face, audioStreamElapsed: audioStreamElapsed)
        }
    }

    // MARK: - Calibration

    public func startCalibration() {
        isCalibrating = true
        calibrationProgress = 0.0
        calibrationAudioBuffers = []
        calibrationEmbeddings = []
        currentCalibrationSentence = 0
        calibrationStartTime = Date()
        DiarizationDebugLog.write("=== startCalibration ===")
    }

    public func startAdditionalCalibration() {
        isCalibrating = true
        calibrationProgress = 0.0
        calibrationAudioBuffers = []
        calibrationStartTime = Date()
    }

    public func stopAdditionalCalibration() {
        guard isCalibrating else { return }
        let allSamples = calibrationAudioBuffers.flatMap { $0 }
        guard !allSamples.isEmpty, let extractor = embeddingExtractor else {
            isCalibrating = false
            return
        }
        do {
            let embedding = try extractor.extract(from: allSamples)
            userEmbeddings.append(embedding)
        } catch {
            NSLog("[GazeSpeakerAttributor] Additional calibration failed: %@", error.localizedDescription)
        }
        isCalibrating = false
        calibrationProgress = 0.0
    }

    public func resetToInitialEmbedding() {
        guard let initial = initialEmbedding else { return }
        userEmbeddings = [initial]
        learningCount = 0
        lastLearningTime = nil
    }

    public func reset() {
        tokenColorMap = [:]
        speakerHistory = []
        jawHistory = []
        gazeHistory = []
        lastTokenAudioTime = 0
        audioBufferQueue = []
    }

    public func deleteEmbedding() {
        try? FileManager.default.removeItem(at: embeddingFileURL)
        userEmbeddings = []
        phase = .calibration
    }

    public func loadSavedEmbedding() {
        guard FileManager.default.fileExists(atPath: embeddingFileURL.path) else { return }
        do {
            let data = try Data(contentsOf: embeddingFileURL)
            let embeddings = try JSONDecoder().decode([[Float]].self, from: data)
            userEmbeddings = embeddings
            initialEmbedding = embeddings.first
            phase = .live
        } catch {
            NSLog("[GazeSpeakerAttributor] Failed to load embeddings: %@", error.localizedDescription)
        }
    }

    // MARK: - Private: Calibration

    private func processCalibrationAudio(_ samples: [Float]) {
        guard isCalibrating, let startTime = calibrationStartTime else { return }
        let elapsed = Date().timeIntervalSince(startTime)
        let totalProgress = (Float(currentCalibrationSentence) + Float(min(elapsed / calibrationDuration, 1.0))) / Float(calibrationSentences.count)
        calibrationProgress = totalProgress
        calibrationAudioBuffers.append(samples)

        // Debug: log every ~1s
        let totalSamples = calibrationAudioBuffers.reduce(0) { $0 + $1.count }
        if Int(elapsed * 10) % 10 == 0 && totalSamples % 16000 < 1600 {
            NSLog("[GazeSpeakerAttributor] calibration buffer: elapsed=%.2fs totalSamples=%d (need ≥1s@16kHz=16000)", elapsed, totalSamples)
            DiarizationDebugLog.write("cal: sentence=\(currentCalibrationSentence) elapsed=\(String(format: "%.2f", elapsed))s totalSamples=\(totalSamples)")
        }

        if elapsed >= calibrationDuration {
            finishCurrentSentence()
        }
    }

    private func finishCurrentSentence() {
        let allSamples = calibrationAudioBuffers.flatMap { $0 }
        NSLog("[GazeSpeakerAttributor] finishCurrentSentence: sentence=%d samples=%d", currentCalibrationSentence, allSamples.count)
        DiarizationDebugLog.write("finishCurrentSentence: sentence=\(currentCalibrationSentence) samples=\(allSamples.count)")
        guard let extractor = embeddingExtractor else {
            NSLog("[GazeSpeakerAttributor] NO EXTRACTOR — bailing")
            DiarizationDebugLog.write("NO EXTRACTOR")
            isCalibrating = false
            return
        }
        do {
            let embedding = try extractor.extract(from: allSamples)
            calibrationEmbeddings.append(embedding)
            currentCalibrationSentence += 1
            NSLog("[GazeSpeakerAttributor] extracted embedding size=%d, advanced to sentence=%d", embedding.count, currentCalibrationSentence)
            DiarizationDebugLog.write("extracted embedding size=\(embedding.count) advanced to sentence=\(currentCalibrationSentence)")
            if currentCalibrationSentence < calibrationSentences.count {
                calibrationAudioBuffers = []
                calibrationStartTime = Date()
                NSLog("[GazeSpeakerAttributor] ready for next sentence")
            } else {
                finishCalibration()
            }
        } catch {
            NSLog("[GazeSpeakerAttributor] Calibration sentence failed: %@", error.localizedDescription)
            DiarizationDebugLog.write("CALIBRATION FAILED: \(error.localizedDescription)")
            isCalibrating = false
        }
    }

    private func finishCalibration() {
        isCalibrating = false
        guard !calibrationEmbeddings.isEmpty else { return }
        userEmbeddings = calibrationEmbeddings
        initialEmbedding = calibrationEmbeddings.first
        learningCount = 0
        lastLearningTime = nil
        phase = .live
        calibrationProgress = 1.0
    }

    // MARK: - Private: Live Recognition

    private func processLiveAudio(_ samples: [Float], face: FaceState, audioStreamElapsed: Double?) {
        guard phase == .live, !userEmbeddings.isEmpty, let extractor = embeddingExtractor else { return }

        // Check face tracking: if jaw hasn't moved in 1s, skip
        if let elapsed = audioStreamElapsed {
            let recentStart = max(0, elapsed - 1.0)
            let recentJawDelta = calculateJawDelta(startTime: recentStart, endTime: elapsed)
            if recentJawDelta < 0.005 {
                speakerMatch = false
                return
            }
        }

        guard samples.count >= 16000 else {
            speakerMatch = false
            return
        }

        let currentEmbedding: [Float]
        do {
            currentEmbedding = try extractor.extract(from: samples)
        } catch {
            speakerMatch = false
            return
        }

        let distances = userEmbeddings.map { cosineSimilarity($0, currentEmbedding) }
        let distance = distances.min() ?? 1.0
        speakerDistance = distance
        speakerMatch = distance < speakerThreshold

        if let elapsed = audioStreamElapsed {
            speakerHistory.append((timestamp: elapsed, distance: distance))
            let cutoff = elapsed - 10.0
            speakerHistory.removeAll { $0.timestamp < cutoff }
        }
    }

    // MARK: - Private: Sensor Queries

    private func querySpeakerAtTime(_ audioTime: Double) -> (Bool, Float) {
        guard !speakerHistory.isEmpty else {
            // No embedding data yet (first ~1s of audio) — return neutral score
            // so jaw/gaze signals can decide without embedding penalty
            return (false, 0.45)
        }
        let closest = speakerHistory.min(by: { abs($0.timestamp - audioTime) < abs($1.timestamp - audioTime) })
        guard let record = closest else { return (false, 0.45) }
        return (record.distance < speakerThreshold, record.distance)
    }

    private func calculateJawDelta(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        let relevant = jawHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        // Fallback: sensor data may lag streaming STT tokens by ~100-300ms.
        // Use the most recent 0.5s of sensor data to estimate activity.
        guard !relevant.isEmpty else {
            let fallback = jawHistory.filter {
                $0.timestamp >= (jawHistory.last?.timestamp ?? 0) - 0.5
            }
            guard fallback.count >= 2 else { return 0 }
            let vals = fallback.map { $0.jawOpen }
            return abs((vals.max() ?? 0) - (vals.min() ?? 0))
        }
        let vals = relevant.map { $0.jawOpen }
        return abs((vals.max() ?? 0) - (vals.min() ?? 0))
    }

    private func calculateJawVelocity(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        var relevant = jawHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        // Fallback: use most recent 0.5s if window has no data
        if relevant.count < 2 {
            relevant = jawHistory.filter {
                $0.timestamp >= (jawHistory.last?.timestamp ?? 0) - 0.5
            }
        }
        guard relevant.count >= 2 else { return 0 }
        var maxVel: Float = 0
        for i in 1..<relevant.count {
            let dt = relevant[i].timestamp - relevant[i-1].timestamp
            guard dt > 0 else { continue }
            let dJaw = abs(relevant[i].jawOpen - relevant[i-1].jawOpen)
            maxVel = max(maxVel, dJaw / Float(dt))
        }
        return maxVel
    }

    private func calculateGazeOnScreenRatio(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        var relevant = gazeHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        if relevant.isEmpty {
            relevant = gazeHistory.filter {
                $0.timestamp >= (gazeHistory.last?.timestamp ?? 0) - 0.5
            }
        }
        guard !relevant.isEmpty else { return 0 }
        return Float(relevant.filter { $0.onScreen }.count) / Float(relevant.count)
    }

    private func calculateMeanHeadYaw(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        var relevant = gazeHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        if relevant.isEmpty {
            relevant = gazeHistory.filter {
                $0.timestamp >= (gazeHistory.last?.timestamp ?? 0) - 0.5
            }
        }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.yaw }.reduce(0, +) / Float(relevant.count)
    }

    private func calculateMeanHeadPitch(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        var relevant = gazeHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        if relevant.isEmpty {
            relevant = gazeHistory.filter {
                $0.timestamp >= (gazeHistory.last?.timestamp ?? 0) - 0.5
            }
        }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.pitch }.reduce(0, +) / Float(relevant.count)
    }

    private func calculateMeanFaceDistance(startTime: Double, endTime: Double) -> Float {
        let s = max(0, startTime - jawMargin)
        let e = endTime + jawMargin
        var relevant = gazeHistory.filter { $0.timestamp >= s && $0.timestamp <= e }
        if relevant.isEmpty {
            relevant = gazeHistory.filter {
                $0.timestamp >= (gazeHistory.last?.timestamp ?? 0) - 0.5
            }
        }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.distance }.reduce(0, +) / Float(relevant.count)
    }

    /// Zone features from jawHistory for streaming-phase scoring.
    /// Returns (jdMean10, jeMean10) — mean jawDelta and mean jawEfficiency
    /// over a ±10s window around `audioTime`.
    private func calculateZoneJawFeatures(around audioTime: Double) -> (jdMean10: Float, jeMean10: Float) {
        let window: Double = 10.0
        let relevant = jawHistory.filter { abs($0.timestamp - audioTime) <= window }
        guard relevant.count >= 2 else { return (0, 0) }
        // Compute per-pair jawDelta and jawEfficiency
        var deltas: [Float] = []
        var efficiencies: [Float] = []
        for i in 1..<relevant.count {
            let dt = relevant[i].timestamp - relevant[i-1].timestamp
            let dJaw = abs(relevant[i].jawOpen - relevant[i-1].jawOpen)
            deltas.append(dJaw)
            if dJaw > 0.001 && dt > 0 {
                let vel = dJaw / Float(dt)
                efficiencies.append(vel / dJaw)
            } else {
                efficiencies.append(0)
            }
        }
        let jdMean = deltas.reduce(0, +) / Float(deltas.count)
        let jeMean = efficiencies.reduce(0, +) / Float(efficiencies.count)
        return (jdMean, jeMean)
    }

    private func makeSpeakerToken(
        text: String, isUserSpeaker: Bool, score: Float,
        audioTime: Double, endTime: Double,
        jawDelta: Float, jawVelocity: Float
    ) -> SpeakerToken {
        var tok = SpeakerToken(
            text: text, isUserSpeaker: isUserSpeaker, score: score,
            audioTime: audioTime, jawDelta: jawDelta, jawVelocity: jawVelocity
        )
        tok.gazeOnScreen = calculateGazeOnScreenRatio(startTime: audioTime, endTime: endTime)
        tok.headYaw = calculateMeanHeadYaw(startTime: audioTime, endTime: endTime)
        tok.headPitch = calculateMeanHeadPitch(startTime: audioTime, endTime: endTime)
        tok.faceDistance = calculateMeanFaceDistance(startTime: audioTime, endTime: endTime)
        return tok
    }

    // MARK: - Private: Token Splitting (v145 unified)

    private func splitTokenBySpeakerChange(_ token: SpeechToken, isFinal: Bool) -> [SpeakerToken] {
        let tokenKey = token.text + "_\(token.startTime)_\(token.endTime)"
        let dt = Float(max(0, token.startTime - lastTokenAudioTime))
        lastTokenAudioTime = token.startTime

        // Zone features: computed once per token from jawHistory (10s window).
        // These are the strongest signals — without them streaming scoring
        // takes an automatic -3.0 penalty from the default-zero path.
        let zone = calculateZoneJawFeatures(around: token.startTime)
        let gazeOn = calculateGazeOnScreenRatio(startTime: token.startTime, endTime: token.endTime)
        let yawAbs = abs(calculateMeanHeadYaw(startTime: token.startTime, endTime: token.endTime))
        let pitchAbs = abs(calculateMeanHeadPitch(startTime: token.startTime, endTime: token.endTime))
        let faceDist = calculateMeanFaceDistance(startTime: token.startTime, endTime: token.endTime)

        // Single char: evaluate directly
        if token.text.count <= 1 {
            let result = querySpeakerAtTime(token.startTime)
            let jd = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let jv = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let jawEff: Float = jd > 0.001 ? jv / jd : 0
            let userScore = calculateUserScore(
                score: result.1, jawDelta: jd, jawVelocity: jv, timeDelta: dt,
                jawEff: jawEff, jdMean10: zone.jdMean10, jeMean10: zone.jeMean10,
                gazeOnScreen: gazeOn, headYawAbs: yawAbs, headPitchAbs: pitchAbs, faceDistance: faceDist
            )
            let isUser = userScore >= perTokenThreshold
            let seg = makeSpeakerToken(
                text: token.text, isUserSpeaker: isUser, score: result.1,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jd, jawVelocity: jv
            )
            if isFinal { tokenColorMap[tokenKey] = (isUser: isUser, score: result.1) }
            return [seg]
        }

        // Multi-char: per-character split + per-char diarization
        let duration = token.endTime - token.startTime
        let charCount = token.text.count
        let timePerChar = charCount > 0 ? duration / Double(charCount) : duration
        var segments: [SpeakerToken] = []
        var currentSpeaker: Bool? = nil
        var currentText = ""
        var currentStartTime = token.startTime

        for (index, char) in token.text.enumerated() {
            let charTime = token.startTime + Double(index) * timePerChar
            let charEndTime = charTime + timePerChar
            let result = querySpeakerAtTime(charTime)
            let jd = calculateJawDelta(startTime: charTime, endTime: charEndTime)
            let jv = calculateJawVelocity(startTime: charTime, endTime: charEndTime)
            let jawEff: Float = jd > 0.001 ? jv / jd : 0
            let userScore = calculateUserScore(
                score: result.1, jawDelta: jd, jawVelocity: jv, timeDelta: dt,
                jawEff: jawEff, jdMean10: zone.jdMean10, jeMean10: zone.jeMean10,
                gazeOnScreen: gazeOn, headYawAbs: yawAbs, headPitchAbs: pitchAbs, faceDistance: faceDist
            )
            let isUser = userScore >= perTokenThreshold

            if currentSpeaker == nil {
                currentSpeaker = isUser
                currentText.append(char)
            } else if currentSpeaker == isUser {
                currentText.append(char)
            } else {
                let segEndTime = charTime
                let segJD = calculateJawDelta(startTime: currentStartTime, endTime: segEndTime)
                let segJV = calculateJawVelocity(startTime: currentStartTime, endTime: segEndTime)
                segments.append(makeSpeakerToken(
                    text: currentText, isUserSpeaker: currentSpeaker!, score: result.1,
                    audioTime: currentStartTime, endTime: segEndTime,
                    jawDelta: segJD, jawVelocity: segJV
                ))
                currentText = String(char)
                currentSpeaker = isUser
                currentStartTime = charTime
            }
        }

        if !currentText.isEmpty, let speaker = currentSpeaker {
            let result = querySpeakerAtTime(currentStartTime)
            let jd = calculateJawDelta(startTime: currentStartTime, endTime: token.endTime)
            let jv = calculateJawVelocity(startTime: currentStartTime, endTime: token.endTime)
            segments.append(makeSpeakerToken(
                text: currentText, isUserSpeaker: speaker, score: result.1,
                audioTime: currentStartTime, endTime: token.endTime,
                jawDelta: jd, jawVelocity: jv
            ))
        }

        if isFinal {
            let majorityUser = segments.filter { $0.isUserSpeaker }.count > segments.count / 2
            let avgScore = segments.isEmpty ? Float(0) : segments.map { $0.score }.reduce(0, +) / Float(segments.count)
            tokenColorMap[tokenKey] = (isUser: majorityUser, score: avgScore)
        }

        return segments
    }

    // MARK: - Private: Smooth Speaker Predictions (v112 zone features)

    private func smoothSpeakerPredictions(_ tokens: [SpeakerToken]) -> [SpeakerToken] {
        guard tokens.count >= 2 else { return tokens }
        let N = tokens.count

        func mean(_ a: [Float]) -> Float {
            guard !a.isEmpty else { return 0 }
            return a.reduce(0, +) / Float(a.count)
        }

        // timeDelta
        var timeDeltaArr = [Float](repeating: 0, count: N)
        for i in 1..<N { timeDeltaArr[i] = Float(max(0, tokens[i].audioTime - tokens[i-1].audioTime)) }

        // jaw efficiency
        let jawEff: [Float] = tokens.map { $0.jawDelta > 0.001 ? $0.jawVelocity / $0.jawDelta : 0 }
        // scoreVelAnti
        let scoreVelAnti: [Float] = tokens.map { (1.0 - $0.score) * $0.jawVelocity }

        // dtZeroRatio (window=5)
        let dtZeroRatio5: [Float] = (0..<N).map { i in
            let lo = max(0, i - 2); let hi = min(N - 1, i + 2)
            var zc: Float = 0
            for j in lo...hi { if timeDeltaArr[j] < 0.001 { zc += 1 } }
            return zc / Float(hi - lo + 1)
        }

        // 10-second window features
        func windowMean(_ i: Int, _ extract: (Int) -> Float) -> Float {
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 { vals.append(extract(j)); j -= 1 }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 { vals.append(extract(j)); j += 1 }
            return mean(vals)
        }

        let jdMean10: [Float] = (0..<N).map { windowMean($0) { tokens[$0].jawDelta } }
        let jeMean10: [Float] = (0..<N).map { windowMean($0) { jawEff[$0] } }
        let scoreMean10: [Float] = (0..<N).map { windowMean($0) { tokens[$0].score } }
        let gazeOnScreen10: [Float] = (0..<N).map { windowMean($0) { tokens[$0].gazeOnScreen } }
        let headYawAbs10: [Float] = (0..<N).map { windowMean($0) { abs(tokens[$0].headYaw) } }
        let headPitchAbs10: [Float] = (0..<N).map { windowMean($0) { abs(tokens[$0].headPitch) } }
        let faceDistance10: [Float] = (0..<N).map { windowMean($0) { tokens[$0].faceDistance } }

        // finalScore estimate
        let finalScoreArr: [Float] = tokens.map { t in
            let jw: Float = (t.jawDelta > 0.05 || t.jawVelocity > 0.3) ? 1.0 : 0.2
            let jawFactor = max(Float(0.1), 1.0 - jw * t.jawDelta)
            let velocityFactor = max(Float(0.1), 1.0 - jw * t.jawVelocity)
            let noMovementFactor: Float = (t.jawDelta < 0.02 && t.jawVelocity < 0.1) ? 1.5 : 1.0
            return t.score * jawFactor * velocityFactor * noMovementFactor
        }

        let isHighJW: [Bool] = tokens.map { $0.jawDelta > 0.05 || $0.jawVelocity > 0.3 }

        // Calculate votes — v16: use per-token gaze/yaw/pitch for gates
        var votes = [Float](repeating: 0, count: N)
        for i in 0..<N {
            votes[i] = calculateUserScore(
                score: tokens[i].score,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity,
                timeDelta: timeDeltaArr[i],
                jawEff: jawEff[i],
                scoreVelAnti: scoreVelAnti[i],
                dtZeroRatio5: dtZeroRatio5[i],
                jdMean10: jdMean10[i],
                jeMean10: jeMean10[i],
                finalScore: finalScoreArr[i],
                isHighJW: isHighJW[i],
                zoneScoreMean: scoreMean10[i],
                gazeOnScreen: tokens[i].gazeOnScreen,     // per-token (was zone)
                headYawAbs: abs(tokens[i].headYaw),       // per-token (was zone)
                headPitchAbs: abs(tokens[i].headPitch),   // per-token (was zone)
                faceDistance: faceDistance10[i]
            )
        }

        // Threshold
        var result = tokens
        for i in 0..<N {
            let isUser = votes[i] >= speakerThreshold
            result[i] = SpeakerToken(
                text: tokens[i].text,
                isUserSpeaker: isUser,
                score: tokens[i].score,
                audioTime: tokens[i].audioTime,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity,
                gazeOnScreen: tokens[i].gazeOnScreen,
                headYaw: tokens[i].headYaw,
                headPitch: tokens[i].headPitch,
                faceDistance: tokens[i].faceDistance
            )
        }

        // v16: Cluster smoothing — group adjacent tokens within 1.5s,
        // borderline tokens inherit cluster-average vote
        let clusterGap: Double = 1.5
        let clusterMargin: Float = 5.0
        var clusters: [[Int]] = []
        var currentCluster: [Int] = [0]
        for i in 1..<N {
            let gap = result[i].audioTime - result[i-1].audioTime
            if gap <= clusterGap {
                currentCluster.append(i)
            } else {
                clusters.append(currentCluster)
                currentCluster = [i]
            }
        }
        clusters.append(currentCluster)

        for cluster in clusters {
            let avgVotes = cluster.map { votes[$0] }.reduce(0, +) / Float(cluster.count)
            let clusterIsUser = avgVotes >= speakerThreshold
            for i in cluster {
                if abs(votes[i] - speakerThreshold) < clusterMargin {
                    if result[i].isUserSpeaker != clusterIsUser {
                        result[i] = SpeakerToken(
                            text: result[i].text,
                            isUserSpeaker: clusterIsUser,
                            score: result[i].score,
                            audioTime: result[i].audioTime,
                            jawDelta: result[i].jawDelta,
                            jawVelocity: result[i].jawVelocity,
                            gazeOnScreen: result[i].gazeOnScreen,
                            headYaw: result[i].headYaw,
                            headPitch: result[i].headPitch,
                            faceDistance: result[i].faceDistance
                        )
                    }
                }
            }
        }

        return result
    }

    // MARK: - Private: User Score (v135b — dual-path: gaze-enhanced + legacy fallback)

    private func calculateUserScore(
        score: Float, jawDelta: Float, jawVelocity: Float, timeDelta: Float = 0,
        jawEff: Float = 0, scoreVelAnti: Float = 0, dtZeroRatio5: Float = 0,
        jdMean10: Float = 0, jeMean10: Float = 0, finalScore: Float = 0,
        isHighJW: Bool = false, zoneScoreMean: Float = 0,
        gazeOnScreen: Float = -1, headYawAbs: Float = 0, headPitchAbs: Float = 0, faceDistance: Float = 0.5
    ) -> Float {
        var votes: Float = 0

        // Determine if we have real gaze data.
        // gazeOnScreen == -1 means caller didn't provide gaze → use legacy path.
        // gazeOnScreen >= 0 means face tracking is active → use gaze-enhanced path.
        let hasGaze = gazeOnScreen >= 0

        if hasGaze {
            // === Gaze-enhanced path (v135b: calibrated from real-world multi-speaker data) ===

            // Gaze: user typically >= 0.4, non-user typically 0
            if gazeOnScreen >= 0.4 { votes += 1.5 }
            else if gazeOnScreen >= 0.2 { votes += 0.5 }
            else if gazeOnScreen < 0.05 { votes -= 1.5 }

            // Head pitch: user 0.35-0.50, non-user 0.55-0.86
            if headPitchAbs > 0.5 { votes -= 2.0 }
            else if headPitchAbs > 0.4 { votes -= 0.5 }
            else { votes += 0.5 }

            // Head yaw
            if headYawAbs > 0.5 { votes -= 1.5 }
            else if headYawAbs > 0.4 { votes -= 0.5 }

            // Distance
            if faceDistance > 1.5 { votes -= 1.5 }
            else if faceDistance > 1.0 { votes -= 0.5 }

            // Embedding score: only penalize when pose confirms non-user
            if score >= 0.7 && gazeOnScreen < 0.2 { votes -= 2.0 }
            else if score >= 0.85 { votes -= 1.0 }
        } else {
            // === Legacy path (no gaze data available) ===
            if finalScore >= 0.7 { votes -= 3.5 }
            if zoneScoreMean >= 0.7 { votes -= 1.5 }
            if score >= 0.55 { votes -= 3.0 }
        }

        // === Physical features (both paths) ===

        // Zone feature: 10s jaw activity (strongest signal)
        if hasGaze {
            if jdMean10 >= 0.03 && jeMean10 >= 5 { votes += 3.5 }
        } else {
            if jdMean10 >= 0.03 && jeMean10 >= 5 { votes += 4.5 }
        }

        // Positive votes
        if jawVelocity >= 0.5 { votes += (hasGaze ? 2.0 : 1.5) }
        else if jawVelocity >= 0.1 { votes += 0.6 }

        if jawDelta >= 0.05 { votes += (hasGaze ? 2.0 : 1.5) }
        else if jawDelta >= 0.02 { votes += (hasGaze ? 0.8 : 0.6) }

        if jawEff >= 5 { votes += 0.5 }
        if scoreVelAnti >= 0.2 { votes += 0.5 }
        if score < 0.45 { votes += 0.5 }
        if timeDelta >= 0.2 { votes += 0.5 }
        if dtZeroRatio5 >= 0.5 { votes += 0.5 }

        // Penalties
        if jdMean10 < 0.005 { votes -= 2.0 }
        if jeMean10 < 1.5 { votes -= 1.0 }

        return votes
    }

    // MARK: - Private: Incremental Learning

    private func tryIncrementalLearningFromTokens(_ tokens: [SpeakerToken]) {
        guard !userEmbeddings.isEmpty, let extractor = embeddingExtractor else { return }

        let userTokens = tokens.filter { token in
            let userScore = calculateUserScore(score: token.score, jawDelta: token.jawDelta, jawVelocity: token.jawVelocity)
            return token.isUserSpeaker && userScore >= learningThreshold && token.jawDelta > 0.05
        }
        guard !userTokens.isEmpty else { return }

        if let lastTime = lastLearningTime {
            guard Date().timeIntervalSince(lastTime) > 5.0 else { return }
        }

        let recentSamples = audioBufferQueue.flatMap { $0 }
        guard recentSamples.count >= 16000 else { return }

        let currentEmbedding: [Float]
        do {
            currentEmbedding = try extractor.extract(from: recentSamples)
        } catch { return }

        // Check similarity to existing embeddings
        let distances = userEmbeddings.map { cosineSimilarity($0, currentEmbedding) }
        guard let minDist = distances.min(), minDist < speakerThreshold else { return }

        // Weighted average with closest embedding
        guard let closestIdx = distances.firstIndex(of: minDist) else { return }
        let oldEmb = userEmbeddings[closestIdx]
        let lr = learningRate
        var newEmb = zip(oldEmb, currentEmbedding).map { $0 * (1 - lr) + $1 * lr }

        // Normalize
        var sumSq: Float = 0
        vDSP_svesq(newEmb, 1, &sumSq, vDSP_Length(newEmb.count))
        let norm = sqrt(sumSq)
        if norm > 0 {
            var d = norm
            vDSP_vsdiv(newEmb, 1, &d, &newEmb, 1, vDSP_Length(newEmb.count))
        }

        userEmbeddings[closestIdx] = newEmb
        learningCount += 1
        lastLearningTime = Date()
    }

    // MARK: - Private: Helpers

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 1.0 }
        let dot = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        guard normA > 0, normB > 0 else { return 1.0 }
        return 1.0 - (dot / (normA * normB))
    }

    // MARK: - Private: Persistence

    private func saveEmbeddings(_ embeddings: [[Float]]) {
        do {
            let data = try JSONEncoder().encode(embeddings)
            try data.write(to: embeddingFileURL)
        } catch {
            NSLog("[GazeSpeakerAttributor] Failed to save embeddings: %@", error.localizedDescription)
        }
    }
}
#endif
