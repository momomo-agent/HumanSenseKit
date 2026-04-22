#if os(iOS)
import Foundation

/// Detects whether the user is speaking by checking co-occurrence of lip movement and audio.
/// Instead of correlating waveforms (Pearson), simply counts how often lip and audio
/// are active at the same time. Immune to time offsets and amplitude differences.
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
        let lipActivity: Float      // absolute value for visualization
        let lipDelta: Float          // frame-to-frame change for detection
        let audioRMS: Float
        let lipActive: Bool
        let audioActive: Bool
    }

    /// Public snapshot of samples for visualization
    public struct SamplePoint: Identifiable {
        public let id: Int
        public let timeOffset: Float  // seconds from window start
        public let lipActivity: Float  // normalized 0-1
        public let audioRMS: Float     // normalized 0-1
    }

    // MARK: - Public state

    /// Co-occurrence ratio [0, 1]. High = user is speaking.
    public private(set) var correlation: Float = 0

    /// Current lip activity value (for debug display).
    public private(set) var lipActivity: Float = 0

    /// Best offset — always 0 for co-occurrence (no offset needed).
    public let bestOffset: Int = 0

    /// Frame counts for debug
    public private(set) var bothCount: Int = 0
    public private(set) var lipOnlyCount: Int = 0
    public private(set) var audioOnlyCount: Int = 0

    /// Whether the user is speaking (lip + audio co-occurring).
    public var isCorrelated: Bool {
        let activeFrames = bothCount + lipOnlyCount + audioOnlyCount
        // Not enough data — default to false (conservative: don't claim user is speaking)
        if activeFrames < minActiveFrames { return false }
        return correlation > cooccurrenceThreshold
    }

    /// Snapshot of current window for visualization (fixed scale).
    public var samplePoints: [SamplePoint] {
        guard let first = samples.first else { return [] }
        let baseTime = first.timestamp
        // Fixed scale: lip 0-3, audio 0-0.05 — no dynamic normalization
        let lipScale: Float = 3.0
        let audioScale: Float = 0.05
        return samples.enumerated().map { i, s in
            SamplePoint(
                id: i,
                timeOffset: Float(s.timestamp - baseTime),
                lipActivity: min(s.lipActivity / lipScale, 1.0),
                audioRMS: min(s.audioRMS / audioScale, 1.0)
            )
        }
    }

    // MARK: - Private state

    private var samples: [Sample] = []
    private var previousActivity: Float = 0
    private let windowDuration: TimeInterval = 1.0
    private let cooccurrenceThreshold: Float = 0.3  // 30% of active frames must be both
    private let minActiveFrames = 8  // need at least ~130ms of activity
    private let lipDeltaThreshold: Float = 0.03  // lip must CHANGE by this much frame-to-frame
    private let audioThreshold: Float = 0.001  // audio RMS above this = "sound present"

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

        // Detect lip MOVEMENT (change), not just position
        let delta = abs(activity - previousActivity)
        previousActivity = activity

        let lipOn = delta > lipDeltaThreshold
        let audioOn = audioRMS > audioThreshold

        samples.append(Sample(
            timestamp: timestamp,
            lipActivity: activity,
            lipDelta: delta,
            audioRMS: audioRMS,
            lipActive: lipOn,
            audioActive: audioOn
        ))

        // Trim old samples
        let cutoff = timestamp - windowDuration
        samples.removeAll { $0.timestamp < cutoff }

        // Recompute co-occurrence
        var both = 0, lipOnly = 0, audioOnly = 0
        for s in samples {
            if s.lipActive && s.audioActive { both += 1 }
            else if s.lipActive { lipOnly += 1 }
            else if s.audioActive { audioOnly += 1 }
        }
        bothCount = both
        lipOnlyCount = lipOnly
        audioOnlyCount = audioOnly

        let activeTotal = both + lipOnly + audioOnly
        if activeTotal >= minActiveFrames {
            correlation = Float(both) / Float(activeTotal)
        } else {
            correlation = 0
        }
    }

    public func reset() {
        samples.removeAll()
        previousActivity = 0
        correlation = 0
        lipActivity = 0
        bothCount = 0
        lipOnlyCount = 0
        audioOnlyCount = 0
    }

    /// Dump current window data for debugging.
    public func dumpWindow() -> String {
        guard let first = samples.first else { return "(empty)" }
        let baseTime = first.timestamp
        var lines = ["t_ms,lip,rms,lipOn,audioOn,both"]
        for s in samples {
            let tMs = Int((s.timestamp - baseTime) * 1000)
            let both = (s.lipActive && s.audioActive) ? 1 : 0
            lines.append(String(format: "%d,%.3f,%.4f,%d,%d,%d",
                                tMs, s.lipActivity, s.audioRMS,
                                s.lipActive ? 1 : 0, s.audioActive ? 1 : 0, both))
        }
        lines.append("# both=\(bothCount) lipOnly=\(lipOnlyCount) audioOnly=\(audioOnlyCount) ratio=\(String(format: "%.2f", correlation))")
        return lines.joined(separator: "\n")
    }
}
#endif
