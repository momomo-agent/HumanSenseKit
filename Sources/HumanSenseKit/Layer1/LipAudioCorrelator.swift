#if os(iOS)
import Foundation

/// Correlates lip movement (jawOpen delta) with audio energy (RMS) over a sliding window.
/// High correlation = user is actually speaking. Low correlation = ambient sound + incidental mouth movement.
@MainActor
public class LipAudioCorrelator {
    public struct Sample {
        let timestamp: TimeInterval
        let jawDelta: Float    // |jawOpen - prevJawOpen|
        let audioRMS: Float    // mic RMS energy
    }

    /// Rolling correlation coefficient [-1, 1]. > 0.3 suggests real speech.
    public private(set) var correlation: Float = 0

    /// Whether lip movement and audio are correlated enough to be real speech.
    /// Returns true when not enough data yet (benefit of the doubt).
    public var isCorrelated: Bool {
        if samples.count < minSamples { return true }  // not enough data, assume user
        return correlation > correlationThreshold
    }

    private var samples: [Sample] = []
    private let windowDuration: TimeInterval = 0.6  // 600ms sliding window
    private let correlationThreshold: Float = 0.25
    private let minSamples = 8  // need enough data points

    public init() {}

    /// Add a new sample. Call this every ARKit frame (~60fps).
    public func addSample(jawDelta: Float, audioRMS: Float, timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        samples.append(Sample(timestamp: timestamp, jawDelta: jawDelta, audioRMS: audioRMS))

        // Trim old samples outside window
        let cutoff = timestamp - windowDuration
        samples.removeAll { $0.timestamp < cutoff }

        // Recompute correlation
        if samples.count >= minSamples {
            correlation = pearsonCorrelation(
                xs: samples.map(\.jawDelta),
                ys: samples.map(\.audioRMS)
            )
        } else {
            correlation = 0
        }
    }

    public func reset() {
        samples.removeAll()
        correlation = 0
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
