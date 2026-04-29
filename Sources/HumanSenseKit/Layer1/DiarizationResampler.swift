#if os(iOS)
import Foundation
import AVFoundation

/// Thread-safe resampler for converting mic buffers to 16kHz mono Float32
/// for the speaker embedding extractor. Used from the audio tap thread,
/// so it's a plain class (no actor isolation) with internal locking.
final class DiarizationResampler: @unchecked Sendable {
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    func resample(_ buffer: AVAudioPCMBuffer) -> [Float]? {
        let inputFormat = buffer.format

        // Fast path: already 16kHz mono Float32
        if inputFormat.sampleRate == 16000,
           inputFormat.channelCount == 1,
           inputFormat.commonFormat == .pcmFormatFloat32,
           let channelData = buffer.floatChannelData {
            let frameLength = Int(buffer.frameLength)
            guard frameLength > 0 else { return nil }
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        }

        lock.lock()
        if targetFormat == nil {
            targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16000,
                                         channels: 1,
                                         interleaved: false)
        }
        guard let target = targetFormat else {
            lock.unlock()
            return nil
        }

        if converter == nil || converter?.inputFormat != inputFormat {
            converter = AVAudioConverter(from: inputFormat, to: target)
        }
        guard let conv = converter else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        // Compute output capacity
        let ratio = target.sampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCapacity) else {
            return nil
        }

        var error: NSError?
        var fed = false
        let status = conv.convert(to: outBuffer, error: &error) { _, inputStatus in
            if fed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return buffer
        }

        guard status != .error, error == nil,
              let channelData = outBuffer.floatChannelData else {
            return nil
        }
        let frameLength = Int(outBuffer.frameLength)
        guard frameLength > 0 else { return nil }
        return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    }
}
#endif
