#if os(iOS)
import Foundation

/// Detects whether the user is speaking by comparing the energy envelopes
/// of lip movement and audio. Instead of frame-level correlation or co-occurrence,
/// extracts low-frequency envelopes (~3-6Hz, matching syllable rate) and correlates those.
///
/// This is robust to:
/// - Frame-level time offsets (envelope is low-freq, 50ms offset doesn't matter)
/// - Micro-jitter and noise (smoothed out by the envelope)
/// - "哈哈哈" (laughing has a 3-5Hz rhythm that shows in both envelopes)
@MainActor
public class LipAudioCorrelator {
    public struct LipFrame {
        let jawOpen: Float
        let mouthClose: Float
        let mouthFunnel: Float
        let mouthPucker: Float
        let mouthLeft: Float
        let mouthRight: Float
        let mouthStretchLeft: Float
        let mouthStretchRight: Float
    }

    /// Public snapshot of samples for visualization
    public struct SamplePoint: Identifiable {
        public let id: Int
        public let timeOffset: Float  // seconds from window start
        public let lipActivity: Float  // normalized 0-1 (envelope)
        public let audioRMS: Float     // normalized 0-1 (envelope)
    }

    // MARK: - Public state

    /// Envelope correlation [-1, 1]. High positive = user is speaking.
    public private(set) var correlation: Float = 0

    /// Current raw lip activity value (for debug display).
    public private(set) var lipActivity: Float = 0

    /// Always 0 for envelope method (no offset search needed).
    public let bestOffset: Int = 0

    /// Frame counts for debug display (reused names for UI compatibility)
    public private(set) var bothCount: Int = 0    // frames where both envelopes are rising
    public private(set) var lipOnlyCount: Int = 0  // lip envelope rising, audio flat
    public private(set) var audioOnlyCount: Int = 0 // audio envelope rising, lip flat

    /// Lip envelope variance (for debug display)
    public private(set) var lipVariance: Float = 0

    /// Whether the user is speaking.
    public var isCorrelated: Bool {
        if rawSamples.count < minSamples { return false }
        // If lip envelope barely moved, it's not the user speaking
        // (micro-jitter in blendshapes can spuriously correlate with audio)
        if lipVariance < 0.0001 { return false }
        return correlation > envelopeThreshold
    }

    /// Snapshot of current window for visualization (envelope values, fixed scale).
    public var samplePoints: [SamplePoint] {
        guard let first = envelopeSamples.first else { return [] }
        let baseTime = first.timestamp
        // Fixed scale: lip envelope 0-0.5 (deviation from baseline), audio envelope 0-0.02
        let lipScale: Float = 0.5
        let audioScale: Float = 0.02
        return envelopeSamples.enumerated().map { i, s in
            SamplePoint(
                id: i,
                timeOffset: Float(s.timestamp - baseTime),
                lipActivity: min(s.lipEnvelope / lipScale, 1.0),
                audioRMS: min(s.audioEnvelope / audioScale, 1.0)
            )
        }
    }

    // MARK: - Private types

    private struct RawSample {
        let timestamp: TimeInterval
        let lipActivity: Float
        let audioRMS: Float
    }

    private struct EnvelopeSample {
        let timestamp: TimeInterval
        let lipEnvelope: Float
        let audioEnvelope: Float
    }

    // MARK: - Private state

    private var rawSamples: [RawSample] = []
    private var envelopeSamples: [EnvelopeSample] = []
    private let windowDuration: TimeInterval = 1.5  // slightly longer window for envelope
    private let envelopeThreshold: Float = 0.45
    private let minSamples = 20  // ~333ms at 60fps
    // Envelope smoothing: EMA with alpha ~0.15 gives ~6Hz cutoff at 60fps
    private let emaAlpha: Float = 0.15
    private var lipEMA: Float = 0
    private var audioEMA: Float = 0
    // Baseline tracking: very slow EMA to track resting lip level
    private let baselineAlpha: Float = 0.005  // ~0.3Hz at 60fps, tracks resting state
    private var lipBaseline: Float = 0
    private var baselineInitialized = false

    public init() {}

    // MARK: - Sampling

    public func addSample(face: LipFrame, audioRMS: Float, timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        let activity = face.jawOpen * 2.0
                     + face.mouthFunnel
                     + face.mouthPucker
                     + face.mouthLeft
                     + face.mouthRight
                     + face.mouthStretchLeft
                     + face.mouthStretchRight
        lipActivity = activity

        // Track resting baseline (very slow EMA)
        if !baselineInitialized {
            lipBaseline = activity
            baselineInitialized = true
        } else {
            lipBaseline = baselineAlpha * activity + (1 - baselineAlpha) * lipBaseline
        }

        // Subtract baseline: only keep the deviation from resting state
        let lipDeviation = max(0, activity - lipBaseline)

        rawSamples.append(RawSample(timestamp: timestamp, lipActivity: activity, audioRMS: audioRMS))

        // Trim old raw samples
        let cutoff = timestamp - windowDuration
        rawSamples.removeAll { $0.timestamp < cutoff }

        // Update EMAs on DEVIATION (not absolute value)
        lipEMA = emaAlpha * lipDeviation + (1 - emaAlpha) * lipEMA
        audioEMA = emaAlpha * audioRMS + (1 - emaAlpha) * audioEMA

        envelopeSamples.append(EnvelopeSample(
            timestamp: timestamp,
            lipEnvelope: lipEMA,
            audioEnvelope: audioEMA
        ))
        envelopeSamples.removeAll { $0.timestamp < cutoff }

        // Compute Pearson correlation on envelopes
        guard envelopeSamples.count >= minSamples else {
            correlation = 0
            return
        }

        let lips = envelopeSamples.map(\.lipEnvelope)
        let audios = envelopeSamples.map(\.audioEnvelope)

        correlation = pearsonCorrelation(xs: lips, ys: audios)

        // Compute lip envelope variance
        let n = Float(lips.count)
        let meanLip = lips.reduce(0, +) / n
        lipVariance = lips.reduce(0) { $0 + ($1 - meanLip) * ($1 - meanLip) } / n

        // Compute debug counts: direction agreement
        var both = 0, lipOnly = 0, audioOnly = 0
        for i in 1..<envelopeSamples.count {
            let lipRising = envelopeSamples[i].lipEnvelope > envelopeSamples[i-1].lipEnvelope + 0.005
            let audioRising = envelopeSamples[i].audioEnvelope > envelopeSamples[i-1].audioEnvelope + 0.0001
            let lipFalling = envelopeSamples[i].lipEnvelope < envelopeSamples[i-1].lipEnvelope - 0.005
            let audioFalling = envelopeSamples[i].audioEnvelope < envelopeSamples[i-1].audioEnvelope - 0.0001

            let lipMoving = lipRising || lipFalling
            let audioMoving = audioRising || audioFalling

            if lipMoving && audioMoving { both += 1 }
            else if lipMoving { lipOnly += 1 }
            else if audioMoving { audioOnly += 1 }
        }
        bothCount = both
        lipOnlyCount = lipOnly
        audioOnlyCount = audioOnly
    }

    public func reset() {
        rawSamples.removeAll()
        envelopeSamples.removeAll()
        lipEMA = 0
        audioEMA = 0
        lipBaseline = 0
        baselineInitialized = false
        correlation = 0
        lipActivity = 0
        lipVariance = 0
        bothCount = 0
        lipOnlyCount = 0
        audioOnlyCount = 0
    }

    /// Dump current window data for debugging.
    public func dumpWindow() -> String {
        guard let first = envelopeSamples.first else { return "(empty)" }
        let baseTime = first.timestamp
        var lines = ["t_ms,lipEnv,audioEnv"]
        for s in envelopeSamples {
            let tMs = Int((s.timestamp - baseTime) * 1000)
            lines.append(String(format: "%d,%.4f,%.5f", tMs, s.lipEnvelope, s.audioEnvelope))
        }
        lines.append("# n=\(envelopeSamples.count) r=\(String(format: "%.3f", correlation)) lipVar=\(String(format: "%.6f", lipVariance)) both=\(bothCount) lipOnly=\(lipOnlyCount) audioOnly=\(audioOnlyCount)")
        return lines.joined(separator: "\n")
    }

    // MARK: - Pearson correlation

    private func pearsonCorrelation(xs: [Float], ys: [Float]) -> Float {
        let n = Float(xs.count)
        guard n > 1 else { return 0 }

        let meanX = xs.reduce(0, +) / n
        let meanY = ys.reduce(0, +) / n

        var sumXY: Float = 0
        var sumX2: Float = 0
        var sumY2: Float = 0

        for i in 0..<xs.count {
            let dx = xs[i] - meanX
            let dy = ys[i] - meanY
            sumXY += dx * dy
            sumX2 += dx * dx
            sumY2 += dy * dy
        }

        let denom = sqrt(sumX2 * sumY2)
        guard denom > 1e-8 else { return 0 }

        return sumXY / denom
    }
}
#endif
