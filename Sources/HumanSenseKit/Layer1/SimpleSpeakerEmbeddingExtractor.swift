#if os(iOS)
import Foundation
import CoreML
import Accelerate

/// Simplified Speaker Embedding extractor for gaze-gated speaker recognition.
/// Uses FBank + Embedding CoreML models to produce 256-dim speaker vectors.
/// Apps must include FBank.mlmodelc and Embedding.mlmodelc in their bundle.
public class SimpleSpeakerEmbeddingExtractor {
    private let fbankModel: MLModel
    private let model: MLModel
    private let audioSampleCount = 160_000
    private let weightFrameCount = 589

    public init() throws {
        guard let fbankURL = Bundle.main.url(
            forResource: "FBank",
            withExtension: "mlmodelc"
        ) else {
            throw ExtractorError.modelNotFound("FBank.mlmodelc")
        }

        guard let embeddingURL = Bundle.main.url(
            forResource: "Embedding",
            withExtension: "mlmodelc"
        ) else {
            throw ExtractorError.modelNotFound("Embedding.mlmodelc")
        }

        let fbankConfig = MLModelConfiguration()
        fbankConfig.computeUnits = .cpuOnly

        let embeddingConfig = MLModelConfiguration()
        embeddingConfig.computeUnits = .cpuAndNeuralEngine

        self.fbankModel = try MLModel(contentsOf: fbankURL, configuration: fbankConfig)
        self.model = try MLModel(contentsOf: embeddingURL, configuration: embeddingConfig)
    }

    /// Extract speaker embedding from audio samples.
    /// - Parameter samples: 16kHz mono Float32 audio, at least 1 second (16000 samples)
    /// - Returns: 256-dim normalized embedding vector
    public func extract(from samples: [Float]) throws -> [Float] {
        guard samples.count >= 16000 else {
            throw ExtractorError.insufficientAudio
        }

        var audioData = samples

        if audioData.count < audioSampleCount {
            repeatPad(&audioData, to: audioSampleCount)
        } else if audioData.count > audioSampleCount {
            audioData = Array(audioData.prefix(audioSampleCount))
        }

        let input = try EmbeddingModelInput(
            fbankFeatures: extractFbankFeatures(from: audioData),
            weights: makeWeights()
        )

        let output = try model.prediction(from: input)

        guard let embeddingFeature = output.featureValue(for: "embedding"),
              let embeddingArray = embeddingFeature.multiArrayValue else {
            throw ExtractorError.invalidOutput
        }

        return normalizedEmbedding(from: embeddingArray)
    }

    private func extractFbankFeatures(from audioData: [Float]) throws -> MLMultiArray {
        let audioArray = try makeFbankAudioInput(from: audioData)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray)
        ])

        let output = try fbankModel.prediction(from: provider)
        guard let features = output.featureValue(for: "fbank_features")?.multiArrayValue else {
            throw ExtractorError.invalidOutput
        }

        return features
    }

    private func makeFbankAudioInput(from audioData: [Float]) throws -> MLMultiArray {
        let shape = fbankModel.modelDescription
            .inputDescriptionsByName["audio"]?
            .multiArrayConstraint?
            .shape
            .filter { $0.intValue > 0 }

        let array = try MLMultiArray(
            shape: shape?.isEmpty == false ? shape! : [1, 1, NSNumber(value: audioSampleCount)],
            dataType: .float32
        )

        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(pointer, 1, vDSP_Length(array.count))

        let copyCount = min(audioData.count, array.count)
        if copyCount > 0 {
            audioData.withUnsafeBufferPointer { buffer in
                vDSP_mmov(
                    buffer.baseAddress!,
                    pointer,
                    vDSP_Length(copyCount),
                    1,
                    vDSP_Length(copyCount),
                    1
                )
            }
        }

        return array
    }

    private func makeWeights() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: weightFrameCount)],
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        var one: Float = 1
        vDSP_vfill(&one, pointer, 1, vDSP_Length(weightFrameCount))
        return array
    }

    private func repeatPad(_ samples: inout [Float], to targetCount: Int) {
        guard !samples.isEmpty else { return }
        while samples.count < targetCount {
            samples.append(contentsOf: samples.prefix(targetCount - samples.count))
        }
    }

    private func normalizedEmbedding(from multiArray: MLMultiArray) -> [Float] {
        let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        var embedding = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count))

        var sumSquares: Float = 0
        vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))

        let norm = sqrt(sumSquares)
        guard norm > 0 else { return embedding }

        var divisor = norm
        vDSP_vsdiv(embedding, 1, &divisor, &embedding, 1, vDSP_Length(embedding.count))
        return embedding
    }

    public enum ExtractorError: Error, LocalizedError {
        case modelNotFound(String)
        case insufficientAudio
        case invalidOutput

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let name):
                return "\(name) not found in bundle"
            case .insufficientAudio:
                return "Audio shorter than 1 second"
            case .invalidOutput:
                return "Invalid model output"
            }
        }
    }
}

/// CoreML model input (manual MLFeatureProvider)
private class EmbeddingModelInput: MLFeatureProvider {
    let fbankFeatures: MLMultiArray
    let weights: MLMultiArray

    var featureNames: Set<String> {
        return ["fbank_features", "weights"]
    }

    init(fbankFeatures: MLMultiArray, weights: MLMultiArray) {
        self.fbankFeatures = fbankFeatures
        self.weights = weights
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "fbank_features":
            return MLFeatureValue(multiArray: fbankFeatures)
        case "weights":
            return MLFeatureValue(multiArray: weights)
        default:
            return nil
        }
    }
}
#endif
