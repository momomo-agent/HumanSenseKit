#if os(iOS)
import Foundation
import AVFoundation
import UIKit

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
        NotificationCenter.default.removeObserver(self, name: UIApplication.willEnterForegroundNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: .AVAudioEngineConfigurationChange, object: nil)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
        // Detect background→foreground stale audio engine.
        // When the app is suspended and resumed, AVAudioEngine often stops
        // silently without firing AVAudioSession.interruptionNotification
        // (those are reserved for active interruptions like phone calls /
        // Siri). Without this hook, we'd only notice via the 5s health
        // timer, leaving audioStreamStartTime stale for up to 5s and
        // producing sampleCount=0 / jaw=0 rows in the reconstructor.
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleWillEnterForeground(_:)),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
        // Detect ARKit / route forced engine config change. The engine is
        // already stopped by the system when this fires.
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleEngineConfigurationChange(_:)),
            name: .AVAudioEngineConfigurationChange,
            object: nil  // engine identity may have changed across restarts
        )
        configureAndStart()
    }
    
    func stop() {
        healthTimer?.invalidate()
        healthTimer = nil
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.interruptionNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: AVAudioSession.routeChangeNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: UIApplication.willEnterForegroundNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: .AVAudioEngineConfigurationChange, object: nil)
        
        if !usesExternalEngine {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        isRunning = false
    }
    
    // MARK: - Public

    /// Re-install the input tap on the audio engine's inputNode.
    /// Call this after the host app restarts an external engine (which
    /// removes all taps via `engine.reset()`). Safe to call multiple
    /// times — removes any existing tap before installing a new one.
    func reinstallTap() {
        let inputNode = audioEngine.inputNode
        inputNode.removeTap(onBus: 0)
        let format = inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0 && format.channelCount > 0 else {
            print("[Audio] reinstallTap: invalid format sr=\(format.sampleRate) ch=\(format.channelCount)")
            return
        }
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { @Sendable [weak self] buffer, when in
            self?.onBuffer?(buffer)
            self?.onBufferWithTime?(buffer, when)
        }
        print("[Audio] Tap re-installed on external engine")
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

    /// App returning to foreground. Check engine health immediately;
    /// without this we'd only catch a stale engine via the 5s health timer.
    @objc private func handleWillEnterForeground(_ notification: Notification) {
        Task { @MainActor in
            guard self.isRunning else { return }
            if self.usesExternalEngine {
                // Host owns the engine; we cannot judge from .isRunning alone
                // because the host may have already restarted it. Always
                // notify so the STT layer re-stamps audioStreamStartTime.
                NSLog("[Audio] willEnterForeground (external engine) — notifying restart")
                self.onRestart?()
                return
            }
            if !self.audioEngine.isRunning {
                NSLog("[Audio] willEnterForeground — engine stale, restarting")
                self.configureAndStart()
                self.onRestart?()
            } else {
                // Engine survived the suspend (background mode active or short suspend).
                // We still notify so STT can re-stamp its timeline against the new
                // sample stream, since AVAudioEngine.isRunning==true does not guarantee
                // the sample timeline survived (especially with VoiceProcessingIO).
                NSLog("[Audio] willEnterForeground — engine still running, notifying anyway")
                self.onRestart?()
            }
        }
    }

    /// Configuration change (route change, ARKit reconfigures audio session,
    /// new sample rate, etc.). The engine is already stopped when this fires.
    @objc private func handleEngineConfigurationChange(_ notification: Notification) {
        Task { @MainActor in
            guard self.isRunning else { return }
            if self.usesExternalEngine {
                NSLog("[Audio] EngineConfigurationChange (external engine) — notifying restart")
                self.onRestart?()
                return
            }
            NSLog("[Audio] EngineConfigurationChange — restarting engine")
            self.configureAndStart()
            self.onRestart?()
        }
    }
}
#endif
