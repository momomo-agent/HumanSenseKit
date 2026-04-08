import Foundation
import AVFoundation
import Combine

@MainActor
class AudioDetectionManager: NSObject, ObservableObject {
    @Published var audioState = AudioState()
    
    private let audioEngine = AVAudioEngine()
    private let speechThreshold: Float = 0.0015  // Lowered to detect quieter speech
    
    func start() {
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }
            let rms = self.calculateRMS(buffer: buffer)
            
            Task { @MainActor in
                self.audioState.volume = rms
                self.audioState.isSpeaking = rms > self.speechThreshold
                
                // Debug output every 30 frames (~1 second)
                if Int.random(in: 0..<30) == 0 {
                    print("DEBUG Audio - volume: \(rms), threshold: \(self.speechThreshold), speaking: \(rms > self.speechThreshold)")
                }
            }
        }
        
        do {
            try audioEngine.start()
        } catch {
            print("Audio engine failed to start: \(error)")
        }
    }
    
    func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
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
        
        let rms = sqrt(sum / Float(frameLength))
        return rms
    }
}
