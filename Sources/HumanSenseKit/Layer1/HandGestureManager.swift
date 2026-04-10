#if os(iOS)
import Foundation
import Vision
import ARKit
import Combine

@MainActor
public class HandGestureManager: NSObject, ObservableObject {
    @Published public var handState = HandState()

    private let processingQueue = DispatchQueue(label: "com.momomo.handgesture", qos: .userInitiated)

    public override init() {
        super.init()
    }

    public func start() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = true
        }
    }

    public func stop() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = false
        }
    }

    private var isProcessingEnabled = false
    private var frameCount = 0
    private var gestureHistory: [HandGesture] = []
    private let historySize = 3  // Require 3 consecutive frames for confirmation

    // Called by FaceTrackingManager with ARFrame
    nonisolated public func processFrame(_ frame: ARFrame) {
        processingQueue.async { [weak self] in
            guard let self = self, self.isProcessingEnabled else { return }

            self.frameCount += 1
            guard self.frameCount % 15 == 0 else { return }  // Process every 15th frame (~4fps)

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
                        self.handState.fingerData = [:]
                        self.gestureHistory.removeAll()
                    }
                    return
                }

                // Process first hand
                let observation = observations[0]
                let chirality = observation.chirality
                let isLeft = (chirality == .left)

                guard let points = try? observation.recognizedPoints(.all) else { return }

                // Use new angle-based gesture recognition
                let gesture = GestureLibrary.match(observation: observation)
                let fingerData = GestureLibrary.analyzer.analyze(observation) ?? [:]

                // Add to history and check for consistency
                self.gestureHistory.append(gesture)
                if self.gestureHistory.count > self.historySize {
                    self.gestureHistory.removeFirst()
                }

                // Only update if gesture is consistent across history
                let confirmedGesture: HandGesture
                if self.gestureHistory.count == self.historySize &&
                   self.gestureHistory.allSatisfy({ $0 == gesture }) {
                    confirmedGesture = gesture
                } else {
                    confirmedGesture = self.handState.gesture  // Keep previous
                }

                Task { @MainActor in
                    self.handState.detected = true
                    self.handState.gesture = confirmedGesture
                    self.handState.fingerData = fingerData
                    self.handState.isLeftHand = isLeft
                    self.handState.handPoints = points.values.map { CGPoint(x: CGFloat($0.location.x), y: CGFloat($0.location.y)) }
                }

            } catch {
                // Hand pose detection error - silently ignore
            }
        }
    }
}
#endif
