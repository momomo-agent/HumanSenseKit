import Foundation
import SwiftUI

/// Main entry point for HumanSenseKit
@MainActor
public class HumanSenseKit {
    public let state: HumanSenseState
    public let observer: HumanSenseObserver
    
    private let faceManager: FaceTrackingManager
    private let audioManager: AudioDetectionManager
    private let handManager: HandGestureManager
    private let engine: HumanStateEngine
    
    public init() {
        self.faceManager = FaceTrackingManager()
        self.audioManager = AudioDetectionManager()
        self.handManager = HandGestureManager()
        self.engine = HumanStateEngine(
            faceManager: faceManager,
            audioManager: audioManager,
            handManager: handManager
        )
        self.state = HumanSenseState(engine: engine)
        self.observer = HumanSenseObserver(state: state)
    }
    
    public func start() {
        engine.start()
    }
    
    public func stop() {
        engine.stop()
    }
    
    // MARK: - Debug UI
    
    #if DEBUG
    public func debugView() -> some View {
        HumanSenseDebugView(kit: self)
    }
    #endif
}
