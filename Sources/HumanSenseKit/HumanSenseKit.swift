import Foundation
import SwiftUI

/// Main entry point for HumanSenseKit
@MainActor
public class HumanSenseKit {
    public let state: HumanSenseState
    public let observer: HumanSenseObserver
    
    private let faceManager: FaceTrackingManager
    private let audioManager: AudioDetectionManager
    private let handManager: HandGestureManager?
    private let engine: HumanStateEngine
    
    public init(enableHandGestures: Bool = false) {
        self.faceManager = FaceTrackingManager()
        self.audioManager = AudioDetectionManager()
        self.handManager = enableHandGestures ? HandGestureManager() : nil
        self.engine = HumanStateEngine(
            faceManager: faceManager,
            audioManager: audioManager,
            handManager: handManager ?? HandGestureManager()
        )
        self.state = HumanSenseState(engine: engine)
        self.observer = HumanSenseObserver(state: state)
    }
    
    public func start() {
        faceManager.start()
        audioManager.start()
        handManager?.start()
    }
    
    public func stop() {
        faceManager.stop()
        audioManager.stop()
        handManager?.stop()
    }
    
    // MARK: - Debug UI
    
    #if DEBUG
    public func debugView() -> some View {
        HumanSenseDebugView(kit: self)
    }
    #endif
}
