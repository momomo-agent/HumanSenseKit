#if os(iOS)
import Foundation

/// Detects whether the user is speaking by comparing the activity envelopes
/// of lip movement and audio using rolling standard deviation.
///
/// Rolling std has NO phase delay (unlike EMA), so lip and audio envelopes
/// are perfectly time-aligned. This fixes the "time mismatch" problem where
/// EMA-smoothed lip and audio signals shift apart.
///
/// The approach:
/// 1. Collect raw lip deviation (from baseline) and audio RMS at 60fps
/// 2. Compute rolling std over a short sub-window (~10 frames, 167ms)
/// 3. Correlate the two std-envelopes over the full 1.5s window
/// 4. High correlation = lip activity variance tracks audio variance = user speaking
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
        public let lipActivity: Float  // normalized 0-1 (rolling std)
        public let audioRMS: Float     // normalized 0-1 (rolling std)
    }

    // MARK: - Public state

    /// Envelope correlation [-1, 1]. High positive = user is speaking.
    public private(set) var correlation: Float = 0

    /// Current raw lip activity value (for debug display).
    public private(set) var lipActivity: Float = 0

    /// Always 0 for this method.
    public let bestOffset: Int = 0

    /// Frame counts for debug display
    public private(set) var bothCount: Int = 0
    public private(set) var lipOnlyCount: Int = 0
    public private(set) var audioOnlyCount: Int = 0

    /// Lip envelope variance (for debug display)
    public private(set) var lipVariance: Float = 0

    /// Timestamp of the most recent frame that passed the correlation gate.
    /// `.distantPast` until the first correlated frame.
    public private(set) var lastCorrelatedAt: TimeInterval = -Double.greatestFiniteMagnitude

    /// Whether the user is speaking (instantaneous).
    public var isCorrelated: Bool {
        if samples.count < minSamples { return false }
        if lipVariance < 0.0001 { return false }
        return correlation > correlationThreshold
    }

    /// Whether the correlator saw a correlated frame within the last `seconds`.
    /// Useful for classifying short utterances whose instantaneous correlation
    /// hasn't warmed up yet but recently did.
    public func wasCorrelated(within seconds: TimeInterval,
                              now: TimeInterval = ProcessInfo.processInfo.systemUptime) -> Bool {
        return (now - lastCorrelatedAt) <= seconds
    }

    /// Snapshot of current window for visualization.
    public var samplePoints: [SamplePoint] {
        let envs = computeEnvelopes()
        guard let first = envs.first else { return [] }
        let baseTime = first.timestamp
        // Fixed scale for visualization
        let lipScale: Float = 0.3
        let audioScale: Float = 0.01
        return envs.enumerated().map { i, e in
            SamplePoint(
                id: i,
                timeOffset: Float(e.timestamp - baseTime),
                lipActivity: min(e.lipStd / lipScale, 1.0),
                audioRMS: min(e.audioStd / audioScale, 1.0)
            )
        }
    }

    // MARK: - Private types

    private struct Sample {
        let timestamp: TimeInterval
        let lipDeviation: Float  // lip activity minus baseline
        let audioRMS: Float
    }

    private struct Envelope {
        let timestamp: TimeInterval
        let lipStd: Float
        let audioStd: Float
    }

    // MARK: - Configuration

    private let windowDuration: TimeInterval = 1.5
    private let correlationThreshold: Float = 0.3  // Pearson correlation threshold
    private let minSamples = 30  // ~500ms at 60fps
    private let subWindowSize = 10  // ~167ms rolling std window
    // Thresholds for "active" in rolling std
    private let lipStdThreshold: Float = 0.02  // lip std above this = mouth is moving
    private let audioStdThreshold: Float = 0.001  // audio std above this = sound present

    // Baseline tracking
    private let baselineAlpha: Float = 0.005
    private var lipBaseline: Float = 0
    private var baselineInitialized = false

    // MARK: - State

    private var samples: [Sample] = []

    /// Ring buffer of (systemUptime, pearson) for token-level attribution.
    /// Keeps the last 5s of per-frame Pearson values so TokenAttributor can
    /// query the correlation at any past moment.
    public struct PearsonSnapshot {
        public let timestamp: TimeInterval  // ProcessInfo.systemUptime
        public let pearson: Float
        public let jawOpen: Float         // raw jaw blendshape
        public let faceVisible: Bool      // was the face detected this frame
    }
    public private(set) var pearsonHistory: [PearsonSnapshot] = []
    private let pearsonHistoryDuration: TimeInterval = 5.0

    /// Per-token feature bundle derived from a time range of history.
    public struct TokenFeatures {
        public let avgPearson: Float      // average Pearson over range
        public let maxPearson: Float      // best Pearson in range
        public let jawStd: Float          // std of jawOpen (mouth moving amount)
        public let jawPeakRate: Float     // local maxima / sec (syllable rate)
        public let faceVisibleRatio: Float // fraction of frames face was detected
        public let sampleCount: Int
    }

    /// Query average Pearson correlation over a time range (systemUptime).
    /// Returns 0 if no samples fall in the range.
    public func averagePearson(from start: TimeInterval, to end: TimeInterval) -> Float {
        let matching = pearsonHistory.filter { $0.timestamp >= start && $0.timestamp <= end }
        guard !matching.isEmpty else { return 0 }
        return matching.map(\.pearson).reduce(0, +) / Float(matching.count)
    }

    /// Compute rich token features over a time range.
    public func features(from start: TimeInterval, to end: TimeInterval) -> TokenFeatures {
        let matching = pearsonHistory.filter { $0.timestamp >= start && $0.timestamp <= end }
        guard !matching.isEmpty else {
            return TokenFeatures(avgPearson: 0, maxPearson: 0, jawStd: 0, jawPeakRate: 0, faceVisibleRatio: 0, sampleCount: 0)
        }
        let pearsons = matching.map(\.pearson)
        let jaws = matching.map(\.jawOpen)
        let visibles = matching.map { $0.faceVisible ? Float(1) : Float(0) }

        // std of jawOpen
        let mean = jaws.reduce(0, +) / Float(jaws.count)
        let variance = jaws.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(jaws.count)
        let std = sqrt(variance)

        // jaw peak count: local maxima above mean + 0.3*std
        var peaks = 0
        if jaws.count >= 3 {
            let threshold = mean + 0.3 * std
            for i in 1..<(jaws.count - 1) {
                if jaws[i] > jaws[i - 1] && jaws[i] >= jaws[i + 1] && jaws[i] > threshold {
                    peaks += 1
                }
            }
        }
        let duration = max(0.05, end - start)
        let peakRate = Float(peaks) / Float(duration)

        return TokenFeatures(
            avgPearson: pearsons.reduce(0, +) / Float(pearsons.count),
            maxPearson: pearsons.max() ?? 0,
            jawStd: std,
            jawPeakRate: peakRate,
            faceVisibleRatio: visibles.reduce(0, +) / Float(visibles.count),
            sampleCount: matching.count
        )
    }

    public init() {}

    // MARK: - Sampling

    public func addSample(face: LipFrame, audioRMS: Float, faceVisible: Bool = true, timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        let activity = face.jawOpen * 2.0
                     + face.mouthFunnel
                     + face.mouthPucker
                     + face.mouthLeft
                     + face.mouthRight
                     + face.mouthStretchLeft
                     + face.mouthStretchRight
        lipActivity = activity

        // Track resting baseline
        if !baselineInitialized {
            lipBaseline = activity
            baselineInitialized = true
        } else {
            lipBaseline = baselineAlpha * activity + (1 - baselineAlpha) * lipBaseline
        }

        let lipDeviation = max(0, activity - lipBaseline)

        samples.append(Sample(timestamp: timestamp, lipDeviation: lipDeviation, audioRMS: audioRMS))

        // Trim old samples
        let cutoff = timestamp - windowDuration
        samples.removeAll { $0.timestamp < cutoff }

        // Need enough samples for rolling std
        guard samples.count >= minSamples else {
            correlation = 0
            lipVariance = 0
            bothCount = 0; lipOnlyCount = 0; audioOnlyCount = 0
            return
        }

        // Compute envelopes and correlate
        let envs = computeEnvelopes()
        guard envs.count >= subWindowSize else {
            correlation = 0
            return
        }

        let lipStds = envs.map(\.lipStd)
        let audioStds = envs.map(\.audioStd)

        // Pearson correlation of lip and audio envelopes.
        // Co-occurrence ratio was unreliable: it measured overlap of "active"
        // frames, not whether the two signals oscillate in sync.
        // Verified on kenefe's 268-frame sample: co-occurrence gave P1=0.48
        // P2=0.76 (inverted!), Pearson gave P1=0.69 P2=0.30 (correct).
        correlation = pearsonCorrelation(xs: lipStds, ys: audioStds)

        // Keep debug counters for visualization
        var both = 0, lipOnly = 0, audioOnly = 0
        for i in 0..<envs.count {
            let lipActive = lipStds[i] > lipStdThreshold
            let audioActive = audioStds[i] > audioStdThreshold
            if lipActive && audioActive { both += 1 }
            else if lipActive { lipOnly += 1 }
            else if audioActive { audioOnly += 1 }
        }
        bothCount = both
        lipOnlyCount = lipOnly
        audioOnlyCount = audioOnly

        // Lip variance (for gate)
        let n = Float(lipStds.count)
        let meanLip = lipStds.reduce(0, +) / n
        lipVariance = lipStds.reduce(0) { $0 + ($1 - meanLip) * ($1 - meanLip) } / n
        lipOnlyCount = lipOnly
        audioOnlyCount = audioOnly

        // Record the most recent moment the full gate passed, so short
        // utterances that reach isCorrelated=true only briefly can still be
        // attributed to the user later when STT finally emits text.
        if lipVariance >= 0.0001 && correlation > correlationThreshold {
            lastCorrelatedAt = timestamp
        }

        // Record Pearson snapshot for token-level attribution
        pearsonHistory.append(PearsonSnapshot(
            timestamp: timestamp,
            pearson: correlation,
            jawOpen: face.jawOpen,
            faceVisible: faceVisible
        ))
        let historyCutoff = timestamp - pearsonHistoryDuration
        while let first = pearsonHistory.first, first.timestamp < historyCutoff {
            pearsonHistory.removeFirst()
        }
    }

    // MARK: - Rolling std envelope

    private func computeEnvelopes() -> [Envelope] {
        guard samples.count >= subWindowSize else { return [] }
        var result: [Envelope] = []
        result.reserveCapacity(samples.count - subWindowSize + 1)

        for i in (subWindowSize - 1)..<samples.count {
            let start = i - subWindowSize + 1
            var lipSum: Float = 0, lipSqSum: Float = 0
            var audioSum: Float = 0, audioSqSum: Float = 0
            let n = Float(subWindowSize)

            for j in start...i {
                let s = samples[j]
                lipSum += s.lipDeviation
                lipSqSum += s.lipDeviation * s.lipDeviation
                audioSum += s.audioRMS
                audioSqSum += s.audioRMS * s.audioRMS
            }

            let lipStd = sqrt(max(0, lipSqSum / n - (lipSum / n) * (lipSum / n)))
            let audioStd = sqrt(max(0, audioSqSum / n - (audioSum / n) * (audioSum / n)))

            result.append(Envelope(
                timestamp: samples[i].timestamp,
                lipStd: lipStd,
                audioStd: audioStd
            ))
        }
        return result
    }

    public func reset() {
        samples.removeAll()
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
        let envs = computeEnvelopes()
        guard let first = envs.first else { return "(empty)" }
        let baseTime = first.timestamp
        var lines = ["t_ms,lipStd,audioStd"]
        for e in envs {
            let tMs = Int((e.timestamp - baseTime) * 1000)
            lines.append(String(format: "%d,%.4f,%.5f", tMs, e.lipStd, e.audioStd))
        }
        lines.append("# n=\(envs.count) r=\(String(format: "%.3f", correlation)) lipVar=\(String(format: "%.6f", lipVariance)) both=\(bothCount) lipOnly=\(lipOnlyCount) audioOnly=\(audioOnlyCount)")
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
