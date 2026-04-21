#if os(iOS)
import AVFoundation

/// Protocol for speech recognition backends.
/// STTManager programs against this interface; concrete implementations
/// are `SpeechAnalyzerBackend` (iOS 26) and `SFSpeechBackend` (legacy).
@MainActor
protocol SpeechRecognitionBackend: AnyObject {
    var onResult: ((_ text: String, _ isFinal: Bool) -> Void)? { get set }
    var onError: ((Error) -> Void)? { get set }
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer)
    func startTask()
    func stop()
    func authorize(completion: @escaping @Sendable (Bool) -> Void)
}
#endif
