#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Uses SpeechTranscriber (preferred for live audio) with volatile results.
/// Falls back to DictationTranscriber if SpeechTranscriber is unavailable.
@MainActor
final class SpeechAnalyzerBackend: SpeechRecognitionBackend {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var detector: SpeechDetector?
    private var resultTask: Task<Void, Never>?
    private var detectorTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var cleanupTask: Task<Void, Never>?
    private var generation: Int = 0

    /// Apple's ML-based voice activity detection result.
    /// Updated in real-time by SpeechDetector.
    private(set) var speechDetected: Bool = false

    /// Callback when speechDetected changes.
    var onSpeechDetected: ((Bool) -> Void)?

    // nonisolated(unsafe) — written on MainActor, read from audio thread
    nonisolated(unsafe) private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    nonisolated(unsafe) private var analyzerFormat: AVAudioFormat?

    /// Contextual strings to improve recognition accuracy.
    /// Set before calling startTask().
    var contextualStrings: [String] = []

    var onResult: ((_ text: String, _ isFinal: Bool, _ speakerLabel: String?, _ audioStartTime: Double?, _ audioEndTime: Double?) -> Void)?
    var onError: ((Error) -> Void)?

    /// Called from audio render thread — must be nonisolated.
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let continuation = inputContinuation else { return }
        guard let format = analyzerFormat else {
            // Format not yet known — drop buffer to avoid sending Float32 to analyzer
            return
        }
        if buffer.format == format {
            continuation.yield(AnalyzerInput(buffer: buffer))
        } else if let converter = AVAudioConverter(from: buffer.format, to: format) {
            // Calculate output frame count with ceiling to avoid underallocation
            let ratio = format.sampleRate / buffer.format.sampleRate
            let frameCount = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio))
            guard frameCount > 0,
                  let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
            var error: NSError?
            converter.convert(to: converted, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if error == nil && converted.frameLength > 0 {
                continuation.yield(AnalyzerInput(buffer: converted))
            }
        } else {
            // Cannot convert — drop buffer rather than crash analyzer
            return
        }
    }
    func startTask() {
        generation += 1
        let myGeneration = generation
        tearDown()

        print("[Speech] startTask gen=\(myGeneration)")

        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: "zh-CN"),
            preset: .timeIndexedProgressiveTranscription
        )
        self.transcriber = transcriber

        let detector = SpeechDetector(
            detectionOptions: .init(sensitivityLevel: .high),
            reportResults: true
        )
        self.detector = detector

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        // Open immediately — AsyncStream buffers until analyzer starts consuming
        self.inputContinuation = continuation

        // Consume transcription results on background thread
        resultTask = Task.detached { [weak self] in
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    let startTime = result.range.start.seconds
                    let endTime = (result.range.start + result.range.duration).seconds
                    print("[Speech] Result gen=\(myGeneration): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    await MainActor.run {
                        guard let self, self.generation == myGeneration else { return }
                        self.onResult?(text, isFinal, nil, startTime, endTime)
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] Error gen=\(myGeneration): \(error.localizedDescription)")
                await MainActor.run {
                    guard let self, self.generation == myGeneration else { return }
                    self.onError?(error)
                }
            }
        }

        // Consume SpeechDetector results (VAD)
        detectorTask = Task.detached { [weak self] in
            do {
                for try await result in detector.results {
                    await MainActor.run {
                        guard let self, self.generation == myGeneration else { return }
                        if self.speechDetected != result.speechDetected {
                            self.speechDetected = result.speechDetected
                            self.onSpeechDetected?(result.speechDetected)
                        }
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] SpeechDetector error gen=\(myGeneration): \(error.localizedDescription)")
            }
        }

        // Start analyzer — get format and start ASAP to minimize latency
        let pendingCleanup = cleanupTask
        let contextStrings = contextualStrings
        analyzerTask = Task.detached { [weak self] in
            await pendingCleanup?.value
            let gen = await self?.generation
            guard gen == myGeneration else { return }
            do {
                // Get optimal format and create analyzer in quick succession
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber, detector])
                guard await self?.generation == myGeneration else { return }

                // Build AnalysisContext with caller-provided contextual strings
                let analysisContext = AnalysisContext()
                if !contextStrings.isEmpty {
                    analysisContext.contextualStrings[.general] = contextStrings
                    print("[Speech] AnalysisContext: \(contextStrings.count) contextual strings")
                }

                // Set format BEFORE creating analyzer so appendBuffer can convert
                await MainActor.run {
                    self?.analyzerFormat = format
                }

                let analyzer = SpeechAnalyzer(
                    inputSequence: stream,
                    modules: [transcriber, detector],
                    analysisContext: analysisContext
                )
                await MainActor.run {
                    self?.analyzer = analyzer
                }

                try await analyzer.prepareToAnalyze(in: format)
                print("[Speech] Analyzer started gen=\(myGeneration)")
            } catch {
                guard !Task.isCancelled else { return }
                guard await self?.generation == myGeneration else { return }
                print("[Speech] Analyzer start failed gen=\(myGeneration): \(error.localizedDescription)")
                await MainActor.run {
                    self?.onError?(error)
                }
            }
        }
    }

    private func tearDown() {
        let oldAnalyzer = analyzer
        let oldContinuation = inputContinuation

        resultTask?.cancel()
        resultTask = nil
        detectorTask?.cancel()
        detectorTask = nil
        analyzerTask?.cancel()
        analyzerTask = nil
        inputContinuation = nil
        analyzer = nil
        transcriber = nil
        detector = nil
        analyzerFormat = nil
        speechDetected = false

        oldContinuation?.finish()

        let previousCleanup = cleanupTask
        cleanupTask = Task {
            await previousCleanup?.value
            if let oldAnalyzer {
                try? await oldAnalyzer.finalizeAndFinishThroughEndOfInput()
            }
        }
    }

    func stop() {
        generation += 1
        tearDown()
    }

    func authorize(completion: @escaping @Sendable (Bool) -> Void) {
        Task {
            let locale = Locale(identifier: "zh-CN")

            // Check if SpeechTranscriber is available on this device
            guard await SpeechTranscriber.isAvailable else {
                print("[Speech] SpeechTranscriber not available on this device")
                completion(false)
                return
            }

            // Check supported locales
            let supported = await SpeechTranscriber.supportedLocales
            guard supported.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) else {
                print("[Speech] zh-CN not supported by SpeechTranscriber")
                completion(false)
                return
            }

            // Download model if needed
            let transcriber = SpeechTranscriber(
                locale: locale,
                preset: .progressiveTranscription
            )
            if let request = try? await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                do {
                    try await request.downloadAndInstall()
                    print("[Speech] Model downloaded")
                } catch {
                    print("[Speech] Model download failed: \(error.localizedDescription)")
                    completion(false)
                    return
                }
            }

            print("[Speech] Authorized and ready")
            completion(true)
        }
    }
}
#endif
