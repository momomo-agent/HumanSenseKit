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
    
    /// The latest ARFrame — use for AvatarKit rendering and camera background.
    public var currentARFrame: ARFrame? {
        engine.currentARFrame
    }
    
    // MARK: - Debug UI
    
    #if DEBUG
    public func debugView() -> some View {
        HumanSenseDebugView(kit: self)
    }
    #endif
}
#endif
