#if os(iOS)
import Foundation
import AVFoundation

/// Layer 0: Pure audio engine lifecycle management.
/// Handles AVAudioEngine start/stop, audio session configuration,
/// interruption recovery, route changes, and health monitoring.
@MainActor
public class AudioEngineManager {
    private var _audioEngine = AVAudioEngine()
    
    /// The audio engine. Can be replaced with an external engine before starting.
    public var audioEngine: AVAudioEngine {
        get { _audioEngine }
        set { _audioEngine = newValue }
    }
    
    /// When true, does NOT configure AVAudioSession or start/stop the engine.
    public var usesExternalEngine = false
    
    /// Called when audio buffers are available.
    /// nonisolated(unsafe) because the audio tap fires on the render thread.
    public nonisolated(unsafe) var onBuffer: ((AVAudioPCMBuffer) -> Void)?
    /// Same as onBuffer but also delivers the AVAudioTime from the tap,
    /// which lets consumers reconstruct the exact host time of the buffer's
    /// first sample. Used for accurate STT↔sensor alignment.
    public nonisolated(unsafe) var onBufferWithTime: ((AVAudioPCMBuffer, AVAudioTime) -> Void)?

    /// Called when the engine needs to be restarted (after interruption/death).
    public nonisolated(unsafe) var onRestart: (() -> Void)?
    
    private(set) var isRunning = false
    private var retryCount: Int = 0
    private let maxRetries: Int = 3
    private let retryDelay: TimeInterval = 0.5
    private var healthTimer: Timer?
    
    func start() {
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.interruptionNotification, object: nil)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
        configureAndStart()
    }
    
    func stop() {
        healthTimer?.invalidate()
        healthTimer = nil
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.interruptionNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.routeChangeNotification, object: nil)
        
        if !usesExternalEngine {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        isRunning = false
    }
    
    // MARK: - Private
    
    private func configureAndStart() {
        if !usesExternalEngine {
            if _audioEngine.isRunning { audioEngine.stop() }
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        
        if !usesExternalEngine {
            let session = AVAudioSession.sharedInstance()
            do {
                if session.category != .playAndRecord {
                    try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers])
                    print("[Audio] Session configured: playAndRecord/voiceChat (VoiceProcessingIO)")
                }
                try session.setActive(true, options: .notifyOthersOnDeactivation)
                try session.overrideOutputAudioPort(.speaker)
            } catch {
                print("[Audio] Session error: \(error.localizedDescription)")
            }
        }
        
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0 && format.channelCount > 0 else {
            print("[Audio] Invalid format: sr=\(format.sampleRate) ch=\(format.channelCount) — retrying")
            retryStart()
            return
        }
        retryCount = 0
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { @Sendable [weak self] buffer, when in
            self?.onBuffer?(buffer)
            self?.onBufferWithTime?(buffer, when)
        }
        
        if !usesExternalEngine {
            audioEngine.prepare()
            do {
                try audioEngine.start()
                print("[Audio] Engine started")
            } catch {
                print("[Audio] Engine start failed: \(error.localizedDescription)")
            }
        }
        isRunning = true
        
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.routeChangeNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(handleRouteChange(_:)), name: AVAudioSession.routeChangeNotification, object: nil)
        
        healthTimer?.invalidate()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if !self.audioEngine.isRunning && self.isRunning && !self.usesExternalEngine {
                    print("[Audio] Engine died — restarting")
                    self.configureAndStart()
                    self.onRestart?()
                }
            }
        }
    }
    
    private func retryStart() {
        retryCount += 1
        guard retryCount <= maxRetries else {
            print("[Audio] Failed after \(maxRetries) retries")
            return
        }
        let attempt = retryCount
        let delay = retryDelay * Double(attempt)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            print("[Audio] Retry \(attempt)/\(self?.maxRetries ?? 0)")
            self?.configureAndStart()
        }
    }
    
    @objc private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }
        switch type {
        case .began:
            print("[Audio] Interrupted")
        case .ended:
            print("[Audio] Interruption ended — restarting")
            Task { @MainActor in
                self.configureAndStart()
                self.onRestart?()
            }
        @unknown default: break
        }
    }
    
    @objc private func handleRouteChange(_ notification: Notification) {
        guard let info = notification.userInfo,
              let reason = info[AVAudioSessionRouteChangeReasonKey] as? UInt else { return }
        let session = AVAudioSession.sharedInstance()
        print("[Audio] Route changed: reason=\(reason), inputs=\(session.currentRoute.inputs.map { "\($0.portName)(\($0.portType.rawValue))" }.joined(separator: ","))")
    }
}
#endif
