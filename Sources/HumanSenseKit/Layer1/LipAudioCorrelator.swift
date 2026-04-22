#if os(iOS)
import Foundation

/// Correlates multi-blendshape lip activity with audio energy over a sliding window.
/// Uses cross-correlation to automatically find the best time offset between lip and audio,
/// then reports the peak Pearson r at that offset.
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

    private struct Sample {
        let timestamp: TimeInterval
        let lipActivity: Float
        let audioRMS: Float
    }

    /// Public snapshot of samples for visualization
    public struct SamplePoint: Identifiable {
        public let id: Int
        public let timeOffset: Float  // seconds from window start
        public let lipActivity: Float
        public let audioRMS: Float
    }

    /// Best correlation coefficient found across all tested offsets [-1, 1].
    public private(set) var correlation: Float = 0

    /// Best offset in frames (positive = lip leads audio). For debug display.
    public private(set) var bestOffset: Int = 0

    /// Current lip activity value (for debug display).
    public private(set) var lipActivity: Float = 0

    /// Whether lip movement and audio are correlated enough to be real speech.
    /// Returns true when not enough data yet (benefit of the doubt).
    public var isCorrelated: Bool {
        if samples.count < minSamples { return true }
        // Lip must actually be moving — if lip variance is too low,
        // any r value is meaningless (flat line correlates with anything)
        let lipValues = samples.map(\.lipActivity)
        let meanLip = lipValues.reduce(0, +) / Float(lipValues.count)
        let lipVariance = lipValues.reduce(0.0) { $0 + ($1 - meanLip) * ($1 - meanLip) } / Float(lipValues.count)
        if lipVariance < 0.01 { return false }  // lip barely moved
        return correlation > correlationThreshold
    }

    /// Snapshot of current window for visualization (normalized 0-1).
    public var samplePoints: [SamplePoint] {
        guard let first = samples.first else { return [] }
        let baseTime = first.timestamp
        let maxLip = samples.map(\.lipActivity).max() ?? 1
        let maxRMS = samples.map(\.audioRMS).max() ?? 1
        let normLip = maxLip > 0.001 ? maxLip : 1
        let normRMS = maxRMS > 0.0001 ? maxRMS : 1
        return samples.enumerated().map { i, s in
            SamplePoint(
                id: i,
                timeOffset: Float(s.timestamp - baseTime),
                lipActivity: s.lipActivity / normLip,
                audioRMS: s.audioRMS / normRMS
            )
        }
    }

    private var samples: [Sample] = []
    private var previousFrame: LipFrame?
    private let windowDuration: TimeInterval = 1.0
    private let correlationThreshold: Float = 0.15
    private let minSamples = 15  // ~250ms at 60fps
    // Cross-correlation search range: -1 to +4 frames (~-17ms to +67ms)
    // Kept tight to avoid overfitting on noise
    private let minLag = -1
    private let maxLag = 4

    public init() {}

    /// Add a new sample every ARKit frame (~60fps).
    public func addSample(face: LipFrame, audioRMS: Float, timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        let activity = face.jawOpen * 2.0
                     + face.mouthFunnel
                     + face.mouthPucker
                     + face.mouthLeft
                     + face.mouthRight
                     + face.mouthStretchLeft
                     + face.mouthStretchRight
        previousFrame = face
        lipActivity = activity

        samples.append(Sample(timestamp: timestamp, lipActivity: activity, audioRMS: audioRMS))

        // Trim old samples outside window
        let cutoff = timestamp - windowDuration
        samples.removeAll { $0.timestamp < cutoff }

        // Need enough samples for cross-correlation
        guard samples.count >= minSamples + maxLag else {
            correlation = 0
            return
        }

        // Filter to active frames for the audio-active check
        let activeCount = samples.filter { $0.audioRMS > 0.001 }.count
        guard activeCount >= minSamples / 2 else {
            // Not enough audio activity — can't determine correlation
            correlation = 0
            return
        }

        // Cross-correlation: try different offsets, find the best Pearson r
        let lip = samples.map(\.lipActivity)
        let audio = samples.map(\.audioRMS)
        let n = lip.count

        var bestR: Float = -2
        var bestLag = 0

        for lag in minLag...maxLag {
            // At lag L: pair lip[i] with audio[i+L]
            // Positive lag means lip is shifted right (lip leads audio)
            let start = max(0, -lag)
            let end = min(n, n - lag)
            guard end - start >= minSamples else { continue }

            var xs: [Float] = []
            var ys: [Float] = []
            for i in start..<end {
                let j = i + lag
                guard j >= 0 && j < n else { continue }
                // Only include frames where at least one signal is active
                if audio[j] > 0.001 || lip[i] > 0.1 {
                    xs.append(lip[i])
                    ys.append(audio[j])
                }
            }

            guard xs.count >= minSamples / 2 else { continue }

            let r = pearsonCorrelation(xs: xs, ys: ys)
            if r > bestR {
                bestR = r
                bestLag = lag
            }
        }

        correlation = bestR > -2 ? bestR : 0
        bestOffset = bestLag
    }

    public func reset() {
        samples.removeAll()
        previousFrame = nil
        correlation = 0
        lipActivity = 0
        bestOffset = 0
    }

    /// Dump current window data for debugging.
    public func dumpWindow() -> String {
        guard let first = samples.first else { return "(empty)" }
        let baseTime = first.timestamp
        var lines = ["t_ms,lip,rms,active"]
        for s in samples {
            let tMs = Int((s.timestamp - baseTime) * 1000)
            let active = s.audioRMS > 0.001 ? 1 : 0
            lines.append(String(format: "%d,%.3f,%.4f,%d", tMs, s.lipActivity, s.audioRMS, active))
        }
        let activeCount = samples.filter { $0.audioRMS > 0.001 }.count
        lines.append("# total=\(samples.count) active=\(activeCount) r=\(String(format: "%.3f", correlation)) offset=\(bestOffset)")
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
