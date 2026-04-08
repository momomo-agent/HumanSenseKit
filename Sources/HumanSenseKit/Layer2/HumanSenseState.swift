import Foundation
import Combine

/// Layer 2: High-level state queries
@MainActor
public class HumanSenseState: ObservableObject {
    @Published public private(set) var rawState: HumanState
    
    private let engine: HumanStateEngine
    private var cancellable: AnyCancellable?
    
    public init(engine: HumanStateEngine) {
        self.engine = engine
        self.rawState = engine.humanState
        
        // Observe engine updates
        cancellable = engine.objectWillChange.sink { [weak self] _ in
            self?.rawState = engine.humanState
        }
    }
    
    // MARK: - High-level queries
    
    public var isPresent: Bool { rawState.face.faceDetected }
    public var isLookingAtScreen: Bool { rawState.face.isLookingAtScreen }
    public var isSpeaking: Bool { rawState.audio.isSpeaking }
    public var isSpeakingToDevice: Bool { isLookingAtScreen && isSpeaking }
    public var eyesClosed: Bool { rawState.face.eyesClosed }
    public var distanceFromCamera: Float { rawState.face.distanceFromCamera }
    public var distanceLabel: String { rawState.face.distanceLabel }
    
    // Head gestures
    public var isNodding: Bool { rawState.face.headGesture == .nodding }
    public var isShakingHead: Bool { rawState.face.headGesture == .shaking }
    public var currentHeadGesture: HeadGesture { rawState.face.headGesture }
    
    // Hand gestures
    public var handDetected: Bool { rawState.hand.detected }
    public var currentHandGesture: HandGesture { rawState.hand.gesture }
    public var isLeftHand: Bool { rawState.hand.isLeftHand }
    public var isRightHand: Bool { !rawState.hand.isLeftHand && rawState.hand.detected }
    
    // Gaze
    public var gazePoint: CGPoint { rawState.face.gazePoint }
    
    // Head orientation (radians)
    public var headYaw: Float { rawState.face.headYaw }
    public var headPitch: Float { rawState.face.headPitch }
    public var headRoll: Float { rawState.face.headRoll }
}
