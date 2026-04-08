import Foundation
import Vision
import AVFoundation
import Combine

@MainActor
class HandGestureManager: NSObject, ObservableObject {
    @Published var handState = HandState()

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.momomo.handgesture", qos: .userInitiated)

    override init() {
        super.init()
        setupCapture()
    }

    private func setupCapture() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device) else { return }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
    }

    func start() {
        Task { @MainActor in
            self.captureSession.startRunning()
        }
    }

    func stop() {
        captureSession.stopRunning()
    }
}

extension HandGestureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .leftMirrored)
        guard let _ = try? handler.perform([request]),
              let observation = request.results?.first else {
            Task { @MainActor in
                self.handState = HandState()
            }
            return
        }

        // Extract hand points for visualization
        let jointNames: [VNHumanHandPoseObservation.JointName] = [
            .wrist, .thumbTip, .thumbIP, .thumbMP, .thumbCMC,
            .indexTip, .indexDIP, .indexPIP, .indexMCP,
            .middleTip, .middleDIP, .middlePIP, .middleMCP,
            .ringTip, .ringDIP, .ringPIP, .ringMCP,
            .littleTip, .littleDIP, .littlePIP, .littleMCP
        ]

        let points = jointNames.compactMap { name -> CGPoint? in
            guard let point = try? observation.recognizedPoint(name),
                  point.confidence > 0.3 else { return nil }
            return CGPoint(x: point.location.x, y: 1 - point.location.y) // flip Y
        }

        let isLeft = observation.chirality == .left
        let gesture = HandGesture.detect(from: observation)

        Task { @MainActor in
            self.handState = HandState(detected: true, gesture: gesture, handPoints: points, isLeftHand: isLeft)
        }
    }
}
