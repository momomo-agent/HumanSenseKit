#if os(iOS)
import Foundation
import CoreMotion
import UIKit
import Combine

@MainActor
public class DeviceMotionManager: ObservableObject {
    @Published public var deviceState = DeviceState()

    private let motionManager = CMMotionManager()
    private let activityManager = CMMotionActivityManager()
    private var accelerationHistory: [Double] = []
    private let historySize = 10
    private var lastWalkingUpdate = Date.distantPast
    private let walkingTimeout: TimeInterval = 3.0

    public init() {}

    public func start() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.1
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
                guard let motion = motion else { return }
                self?.deviceState.gravity = (motion.gravity.x, motion.gravity.y, motion.gravity.z)
                self?.updatePosture(from: motion)
                self?.updateHoldingState(from: motion)
            }
        }

        if CMMotionActivityManager.isActivityAvailable() {
            activityManager.startActivityUpdates(to: .main) { [weak self] activity in
                guard let activity = activity else { return }
                self?.deviceState.isWalking = activity.walking
                if activity.walking {
                    self?.lastWalkingUpdate = Date()
                }
            }
        }

        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            if self.deviceState.isWalking && Date().timeIntervalSince(self.lastWalkingUpdate) > self.walkingTimeout {
                self.deviceState.isWalking = false
            }
        }

        NotificationCenter.default.addObserver(
            forName: UIDevice.orientationDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateOrientation()
        }

        updateOrientation()
    }

    public func stop() {
        motionManager.stopDeviceMotionUpdates()
        activityManager.stopActivityUpdates()
        NotificationCenter.default.removeObserver(self)
    }

    private func updatePosture(from motion: CMDeviceMotion) {
        let gravity = motion.gravity
        let absY = abs(gravity.y)
        let absZ = abs(gravity.z)

        if deviceState.isWalking {
            deviceState.posture = .holdingWalking
            return
        }

        if absZ > 0.8 {
            deviceState.posture = gravity.z < 0 ? .faceUp : .faceDown
        } else if absY > 0.7 {
            deviceState.posture = .uprightStand
        } else {
            deviceState.posture = .landscapeStand
        }
    }

    private func updateOrientation() {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene else {
            deviceState.orientation = .unknown
            return
        }

        switch windowScene.interfaceOrientation {
        case .portrait, .portraitUpsideDown:
            deviceState.orientation = .portrait
        case .landscapeLeft, .landscapeRight:
            deviceState.orientation = .landscape
        default:
            deviceState.orientation = .unknown
        }
    }

    private func updateHoldingState(from motion: CMDeviceMotion) {
        let userAccel = motion.userAcceleration
        let magnitude = sqrt(userAccel.x * userAccel.x +
                           userAccel.y * userAccel.y +
                           userAccel.z * userAccel.z)

        accelerationHistory.append(magnitude)
        if accelerationHistory.count > historySize {
            accelerationHistory.removeFirst()
        }

        guard accelerationHistory.count == historySize else { return }
        let mean = accelerationHistory.reduce(0, +) / Double(historySize)
        let variance = accelerationHistory.map { pow($0 - mean, 2) }.reduce(0, +) / Double(historySize)

        deviceState.isHolding = variance > 0.001
    }
}
#endif
