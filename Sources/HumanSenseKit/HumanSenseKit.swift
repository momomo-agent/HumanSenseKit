#if os(iOS)
import Foundation
import SwiftUI
import ARKit

/// Main entry point for HumanSenseKit
@MainActor
public class HumanSenseKit {
    public let state: HumanSenseState
    public let observer: HumanSenseObserver
    public let sttManager: STTManager
    
    private let engine: HumanStateEngine
    
    public init(enableHandGestures: Bool = false, enableSTT: Bool = true) {
        self.engine = HumanStateEngine()
        self.sttManager = engine.sttManager
        self.state = HumanSenseState(engine: engine)
        self.observer = HumanSenseObserver(state: state)
    }
    
    /// Real-time ARFrame callback — bypasses SwiftUI for 60fps consumers (e.g. AvatarKit).
    /// WARNING: Do NOT retain the ARFrame - extract needed data immediately.
    public var onARFrame: ((ARFrame) -> Void)? {
        get { engine.onARFrame }
        set { engine.onARFrame = newValue }
    }

    /// Prefer a wide-angle (ultra-wide / Center Stage) front-camera
    /// video format for ARKit face tracking when available. Defaults
    /// to `true` — on iPhone 15 Pro / 17 Pro this selects a much
    /// wider FOV than the ARKit default. Silently ignored on devices
    /// that don't advertise an ultra-wide ARVideoFormat.
    ///
    /// Must be set before `start(session:)` for it to take effect on
    /// the initial configuration; changing it mid-session requires a
    /// stop/start cycle.
    public var preferWideAngleCamera: Bool {
        get { engine.faceManager.preferWideAngle }
        set { engine.faceManager.preferWideAngle = newValue }
    }

    /// Start sensing. Optionally provide a shared ARSession.
    /// If nil, HumanSenseKit creates its own.
    public func start(session: ARSession? = nil) {
        engine.start(session: session)
    }

    public func stop() {
        engine.stop()
    }

    // MARK: - Raw Sensor Access

    /// The current ARFaceAnchor from face tracking, if available.
    public var currentFaceAnchor: ARFaceAnchor? {
        engine.currentFaceAnchor
    }

    /// Lip-audio correlation value (0-1). Higher = lips and audio are in sync.
    public var lipCorrelation: Float {
        engine.lipAudioCorrelator.correlation
    }

    /// Whether lip-audio correlation passes the threshold.
    public var lipCorrelated: Bool {
        engine.lipAudioCorrelator.isCorrelated
    }
    
    // MARK: - Debug UI
    
    #if DEBUG
    public func debugView() -> some View {
        HumanSenseDebugView(kit: self)
    }
    #endif
}
#endif
