#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using SFSpeechRecognizer (stable, pre-iOS 26).
///
/// SpeechAnalyzer (iOS 26) has dispatch_assert_queue crashes in beta.
/// Falling back to SFSpeechRecognizer until iOS 26 stabilizes.
@MainActor
public class SpeechRecognitionManager {
    private var recognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?
    public var onError: ((Error) -> Void)?

    /// Append audio buffer to the recognition request.
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }

    /// Start recognition. Safe to call multiple times.
    func startTask() {
        stopRecognition()

        print("[Speech] startTask (SFSpeechRecognizer)")

        let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
        self.recognizer = recognizer

        guard let recognizer, recognizer.isAvailable else {
            print("[Speech] Recognizer not available")
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        self.recognitionRequest = request

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    let text = result.bestTranscription.formattedString
                    let isFinal = result.isFinal
                    print("[Speech] '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    self.onResult?(text, isFinal)
                    if isFinal {
                        self.stopRecognition()
                    }
                }
                if let error {
                    print("[Speech] Error: \(error.localizedDescription)")
                    self.onError?(error)
                    self.stopRecognition()
                }
            }
        }
    }

    /// Stop recognition and clean up.
    func stop() {
        stopRecognition()
    }

    private func stopRecognition() {
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        recognizer = nil
    }

    /// Request authorization.
    func authorize(completion: @escaping @Sendable (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            completion(status == .authorized)
        }
    }
}
#endif
