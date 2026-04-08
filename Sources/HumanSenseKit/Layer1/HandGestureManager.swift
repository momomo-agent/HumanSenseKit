import Foundation
import Vision
import ARKit
import Combine

@MainActor
class HandGestureManager: NSObject, ObservableObject {
    @Published var handState = HandState()

    private let processingQueue = DispatchQueue(label: "com.momomo.handgesture", qos: .userInitiated)

    func start() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = true
        }
    }

    func stop() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = false
        }
    }
    
    private var isProcessingEnabled = false
    private var frameCount = 0
    
    // Called by FaceTrackingManager with ARFrame
    nonisolated func processFrame(_ frame: ARFrame) {
        processingQueue.async { [weak self] in
            guard let self = self, self.isProcessingEnabled else { return }
            
            self.frameCount += 1
            guard self.frameCount % 10 == 0 else { return }  // Process every 10th frame
            
            let pixelBuffer = frame.capturedImage
            
            let request = VNDetectHumanHandPoseRequest()
            request.maximumHandCount = 2
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
            
            do {
                try handler.perform([request])
                
                guard let observations = request.results, !observations.isEmpty else {
                    Task { @MainActor in
                        self.handState.detected = false
                        self.handState.gesture = .none
                    }
                    return
                }
                
                // Process first hand
                let observation = observations[0]
                let chirality = observation.chirality
                let isLeft = (chirality == .left)
                
                guard let points = try? observation.recognizedPoints(.all) else { return }
                
                let gesture = self.recognizeGesture(from: points)
                
                Task { @MainActor in
                    self.handState.detected = true
                    self.handState.gesture = gesture
                    self.handState.isLeftHand = isLeft
                    self.handState.handPoints = points.values.map { CGPoint(x: CGFloat($0.location.x), y: CGFloat($0.location.y)) }
                }
                
            } catch {
                print("Hand pose detection error: \(error)")
            }
        }
    }
    
    private func recognizeGesture(from points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]) -> HandGesture {
        guard let thumb = points[.thumbTip],
              let index = points[.indexTip],
              let middle = points[.middleTip],
              let ring = points[.ringTip],
              let pinky = points[.littleTip],
              let wrist = points[.wrist],
              thumb.confidence > 0.3,
              index.confidence > 0.3 else {
            return .none
        }
        
        let thumbUp = thumb.location.y > wrist.location.y + 0.15
        let indexUp = index.location.y > wrist.location.y + 0.15
        let middleUp = middle.location.y > wrist.location.y + 0.15
        let ringDown = ring.location.y < wrist.location.y + 0.1
        let pinkyDown = pinky.location.y < wrist.location.y + 0.1
        
        if thumbUp && !indexUp && !middleUp && ringDown && pinkyDown {
            return .thumbsUp
        }
        
        if indexUp && middleUp && !thumbUp && ringDown && pinkyDown {
            return .peace
        }
        
        if thumbUp && indexUp && middleUp && middle.confidence > 0.3 && ring.confidence > 0.3 {
            return .openPalm
        }
        
        if !thumbUp && !indexUp && !middleUp && ringDown && pinkyDown {
            return .fist
        }
        
        if indexUp && !middleUp && !thumbUp && ringDown && pinkyDown {
            return .pointing
        }
        
        return .none
    }
}
