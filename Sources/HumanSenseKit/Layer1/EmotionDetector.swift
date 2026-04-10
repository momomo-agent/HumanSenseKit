#if os(iOS)
import Foundation
import ARKit

public class EmotionDetector {
    public init() {}

    public func detectEmotion(from blendShapes: [ARFaceAnchor.BlendShapeLocation: NSNumber], isSpeaking: Bool = false) -> Emotion {
        // Extract relevant blend shapes
        let smileLeft = blendShapes[.mouthSmileLeft]?.floatValue ?? 0
        let smileRight = blendShapes[.mouthSmileRight]?.floatValue ?? 0
        let frownLeft = blendShapes[.mouthFrownLeft]?.floatValue ?? 0
        let frownRight = blendShapes[.mouthFrownRight]?.floatValue ?? 0
        let browInnerUp = blendShapes[.browInnerUp]?.floatValue ?? 0
        let browDownLeft = blendShapes[.browDownLeft]?.floatValue ?? 0
        let browDownRight = blendShapes[.browDownRight]?.floatValue ?? 0
        let jawOpen = blendShapes[.jawOpen]?.floatValue ?? 0
        let noseSneerLeft = blendShapes[.noseSneerLeft]?.floatValue ?? 0
        let noseSneerRight = blendShapes[.noseSneerRight]?.floatValue ?? 0

        // If speaking, don't detect surprise (mouth open is expected)
        if isSpeaking {
            let smileScore = (smileLeft + smileRight) / 2
            if smileScore > 0.3 { return .happy }
            return .neutral
        }

        // Calculate emotion scores
        let smileScore = (smileLeft + smileRight) / 2
        let frownScore = (frownLeft + frownRight) / 2
        let surpriseScore = browInnerUp * 0.7 + jawOpen * 0.3
        let angryScore = (browDownLeft + browDownRight) / 2
        let disgustScore = (noseSneerLeft + noseSneerRight) / 2

        // Determine dominant emotion
        let scores: [(Emotion, Float)] = [
            (.happy, smileScore),
            (.sad, frownScore),
            (.surprised, surpriseScore),
            (.angry, angryScore),
            (.disgusted, disgustScore)
        ]

        // Find max score
        if let maxEmotion = scores.max(by: { $0.1 < $1.1 }), maxEmotion.1 > 0.3 {
            return maxEmotion.0
        }

        return .neutral
    }
}
#endif
