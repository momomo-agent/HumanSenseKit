#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Legacy speech recognition using SFSpeechRecognizer.
///
/// Fallback backend for devices where SpeechAnalyzer / SpeechTranscriber
/// is unavailable. Handles the 50-second task duration limit with auto-restart.
@MainActor
final class SFSpeechBackend: SpeechRecognitionBackend {
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    // nonisolated(unsafe) — written on MainActor, read from audio thread
    nonisolated(unsafe) private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var taskGeneration: Int = 0
    private var taskDurationTimer: Timer?
    private let maxTaskDuration: TimeInterval = 50.0

    var onResult: ((_ text: String, _ isFinal: Bool, _ speakerLabel: String?, _ audioStartTime: Double?, _ audioEndTime: Double?) -> Void)?
    var onTokens: ((_ tokens: [SpeechToken], _ isFinal: Bool) -> Void)?
    var onFirstBuffer: ((Date) -> Void)?
    var onError: ((Error) -> Void)?

    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }

    nonisolated func noteBufferTime(buffer: AVAudioPCMBuffer, when: AVAudioTime) {
        // SFSpeechRecognizer doesn't expose per-token audio time ranges,
        // so we don't need precise origin stamping here.
    }

    func startTask() {
        taskGeneration += 1
        let gen = taskGeneration

        resetTaskDurationTimer()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        recognitionRequest = request

        print("[SFSpeech] startTask gen=\(gen), onDevice=\(request.requiresOnDeviceRecognition ? 1 : 0)")

        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                guard gen == self.taskGeneration else { return }

                if let result {
                    let text = result.bestTranscription.formattedString
                    let isFinal = result.isFinal
                    print("[SFSpeech] Result gen=\(gen): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    self.onResult?(text, isFinal, nil, nil, nil)
                    if isFinal {
                        self.startTask()
                    }
                }

                if let error, !(result?.isFinal ?? false) {
                    print("[SFSpeech] Error gen=\(gen): \(error.localizedDescription)")
                    self.onError?(error)
                    self.taskGeneration += 1
                    self.startTask()
                }
            }
        }
    }

    func stop() {
        taskGeneration += 1
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
    }

    func authorize(completion: @escaping @Sendable (Bool) -> Void) {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            print("[SFSpeech] Recognizer not available")
            completion(false)
            return
        }
        SFSpeechRecognizer.requestAuthorization { status in
            print("[SFSpeech] Auth status: \(status.rawValue)")
            completion(status == .authorized)
        }
    }

    // MARK: - Private

    private func resetTaskDurationTimer() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                print("[SFSpeech] Task duration limit — restarting")
                self.startTask()
            }
        }
    }
}
#endif
