import Foundation
import Speech
import AVFoundation
import SwiftUI

public struct SpeechSegment: Identifiable {
    public let id = UUID()
    public let text: String
    public let isToScreen: Bool
    
    public init(text: String, isToScreen: Bool) {
        self.text = text
        self.isToScreen = isToScreen
    }
}

@MainActor
public class STTManager: NSObject, ObservableObject {
    @Published public var segments: [SpeechSegment] = []
    @Published public var isListening: Bool = false
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var lastText: String = ""
    public var isLookingAtScreen: Bool = false  // Set by external observer
    
    public override init() {
        super.init()
    }
    
    public func start() {
        print("STT: start() called")
        
        // Check if recognizer is available
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            print("STT: Speech recognizer not available")
            return
        }
        
        print("STT: Speech recognizer available, requesting authorization...")
        
        // Request authorization
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            print("STT Authorization status: \(status.rawValue)")
            guard status == .authorized else {
                print("Speech recognition not authorized")
                return
            }
            
            Task { @MainActor in
                print("Starting STT recognition...")
                self?.startRecognition()
            }
        }
    }
    
    public func stop() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        isListening = false
    }
    
    private func startRecognition() {
        print("STT: Starting recognition task...")
        
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { 
            print("STT: Failed to create recognition request")
            return 
        }
        recognitionRequest.shouldReportPartialResults = true
        
        print("STT: Created recognition request")
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let error = error {
                print("STT: Recognition error: \(error)")
            }
            
            if let result = result {
                Task { @MainActor in
                    let newText = result.bestTranscription.formattedString
                    print("STT: Recognized text: \(newText)")
                    
                    // Check if new text was added
                    if newText.count > self.lastText.count {
                        let addedText = String(newText.dropFirst(self.lastText.count))
                        if !addedText.trimmingCharacters(in: .whitespaces).isEmpty {
                            self.segments.append(SpeechSegment(
                                text: addedText,
                                isToScreen: self.isLookingAtScreen
                            ))
                        }
                    }
                    
                    self.lastText = newText
                }
            }
            
            if error != nil || result?.isFinal == true {
                print("STT: Recognition ended (error: \(error != nil), isFinal: \(result?.isFinal == true))")
                self.audioEngine.stop()
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                
                // Restart recognition after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    if self.isListening {
                        print("STT: Restarting recognition...")
                        self.startRecognition()
                    }
                }
            }
        }
        
        print("STT: Recognition task started")
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // Start audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        print("STT: Audio engine started, isListening = true")
    }
}
