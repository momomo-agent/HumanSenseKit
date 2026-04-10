#if os(iOS)
import Foundation
import AVFoundation
import Combine

/// Pure audio-level analysis — no longer owns an AudioEngine.
/// Receives buffers from STTManager via `processBuffer(_:)`.
@MainActor
public class AudioDetectionManager: NSObject, ObservableObject {
    @Published public var audioState = AudioState()

    private let speechThreshold: Float = 0.0008
    private var silenceTimer: Timer?
    private let silenceDelay: TimeInterval = 0.5

    public override init() {
        super.init()
    }

    /// Called from STTManager's shared audio tap.
    public func processBuffer(_ buffer: AVAudioPCMBuffer) {
        let rms = calculateRMS(buffer: buffer)

        audioState.volume = rms
        let aboveThreshold = rms > speechThreshold

        if aboveThreshold {
            silenceTimer?.invalidate()
            silenceTimer = nil
            audioState.isSpeaking = true
        } else if audioState.isSpeaking && silenceTimer == nil {
            silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceDelay, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    self?.audioState.isSpeaking = false
                    self?.silenceTimer = nil
                }
            }
        }
    }

    public func stop() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        audioState.isSpeaking = false
    }

    private func calculateRMS(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else { return 0 }
        let channelDataValue = channelData.pointee
        let frameLength = Int(buffer.frameLength)

        var sum: Float = 0
        for i in 0..<frameLength {
            let sample = channelDataValue[i]
            sum += sample * sample
        }

        return sqrt(sum / Float(frameLength))
    }
}
#endif
