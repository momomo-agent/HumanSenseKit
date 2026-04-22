#if os(iOS)
import Foundation

/// Correlates multi-blendshape lip activity with audio energy over a sliding window.
/// Uses sum of frame-to-frame deltas across 8 lip blendshapes as "lip energy",
/// then Pearson-correlates with audio RMS. High correlation = real speech.
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
        let lipActivity: Float  // sum of |delta| across all lip blendshapes
        let audioRMS: Float
    }

    /// Rolling correlation coefficient [-1, 1].
    public private(set) var correlation: Float = 0

    /// Current lip activity value (for debug display).
    public private(set) var lipActivity: Float = 0

    /// Whether lip movement and audio are correlated enough to be real speech.
    /// Returns true when not enough data yet (benefit of the doubt).
    public var isCorrelated: Bool {
        if samples.count < minSamples { return true }
        return correlation > correlationThreshold
    }

    private var samples: [Sample] = []
    private var previousFrame: LipFrame?
    private let windowDuration: TimeInterval = 1.0  // 1s sliding window (was 600ms)
    private let correlationThreshold: Float = 0.15
    private let minSamples = 15  // ~250ms at 60fps

    public init() {}

    /// Add a new sample every ARKit frame (~60fps).
    public func addSample(face: LipFrame, audioRMS: Float, timestamp: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        // Compute lip activity = weighted sum of mouth blendshape values
        // Higher when mouth is actively moving/open during speech
        let activity = face.jawOpen * 2.0  // jaw is the strongest signal
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

        // Recompute correlation — only on frames where audio is active
        // Silent frames dilute the correlation (both signals are ~constant)
        let activeFrames = samples.filter { $0.audioRMS > 0.001 }
        if activeFrames.count >= minSamples {
            correlation = pearsonCorrelation(
                xs: activeFrames.map(\.lipActivity),
                ys: activeFrames.map(\.audioRMS)
            )
        } else if samples.count >= minSamples {
            // Not enough active frames — use all samples but lower confidence
            correlation = pearsonCorrelation(
                xs: samples.map(\.lipActivity),
                ys: samples.map(\.audioRMS)
            )
        } else {
            correlation = 0
        }
    }

    public func reset() {
        samples.removeAll()
        previousFrame = nil
        correlation = 0
        lipActivity = 0
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
