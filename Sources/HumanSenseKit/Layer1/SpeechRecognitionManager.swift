#if os(iOS)
import Foundation
import Speech

/// Layer 1: Speech recognition task management.
/// Handles SFSpeechRecognizer lifecycle, request creation, task binding,
/// and the 50s task duration limit. Outputs raw recognition results.
@MainActor
public class SpeechRecognitionManager {
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var taskGeneration: Int = 0
    private var taskDurationTimer: Timer?
    private let maxTaskDuration: TimeInterval = 50.0
    
    /// Called with each recognition result (partial or final).
    public var onResult: ((SFSpeechRecognitionResult) -> Void)?
    
    /// Called when a recognition error occurs.
    public var onError: ((Error) -> Void)?
    
    /// Called when the task needs to split (duration limit reached).
    public var onTaskSplit: (() -> Void)?
    
    /// Append audio buffer to the current recognition request.
    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
    }
    
    /// Start a new recognition task. Cancels any existing one.
    func startTask() {
        taskGeneration += 1
        let gen = taskGeneration
        
        resetTaskDurationTimer()
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        recognitionRequest = request
        
        print("[Speech] startTask gen=\(gen), onDevice=\(request.requiresOnDeviceRecognition ? 1 : 0)")
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                guard gen == self.taskGeneration else { return }
                
                if let result {
                    print("[Speech] Result: '\(result.bestTranscription.formattedString.prefix(60))' isFinal=\(result.isFinal ? 1 : 0)")
                    self.onResult?(result)
                    if result.isFinal {
                        self.startTask()
                    }
                }
                
                if let error, !(result?.isFinal ?? false) {
                    print("[Speech] Error: \(error.localizedDescription) (code=\((error as NSError).code))")
                    self.onError?(error)
                    self.taskGeneration += 1
                    self.startTask()
                }
            }
        }
    }
    
    /// End the current task and start a fresh one (for sentence splitting).
    func splitTask() {
        let oldRequest = recognitionRequest
        let oldTask = recognitionTask
        
        taskGeneration += 1
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        recognitionRequest = request
        
        oldRequest?.endAudio()
        oldTask?.cancel()
        
        resetTaskDurationTimer()
        
        let gen = taskGeneration
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                guard gen == self.taskGeneration else { return }
                if let result {
                    print("[Speech] Result: '\(result.bestTranscription.formattedString.prefix(60))' isFinal=\(result.isFinal ? 1 : 0)")
                    self.onResult?(result)
                    if result.isFinal {
                        self.startTask()
                    }
                }
                if let error, !(result?.isFinal ?? false) {
                    print("[Speech] Error: \(error.localizedDescription)")
                    self.onError?(error)
                    self.taskGeneration += 1
                    self.startTask()
                }
            }
        }
    }
    
    /// Stop recognition entirely.
    func stop() {
        taskGeneration += 1
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
    }
    
    /// Check authorization and call completion when ready.
    func authorize(completion: @escaping (Bool) -> Void) {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            print("[Speech] Recognizer not available")
            completion(false)
            return
        }
        SFSpeechRecognizer.requestAuthorization { status in
            print("[Speech] Auth status: \(status.rawValue)")
            completion(status == .authorized)
        }
    }
    
    private func resetTaskDurationTimer() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.onTaskSplit?()
            }
        }
    }
}
#endif
