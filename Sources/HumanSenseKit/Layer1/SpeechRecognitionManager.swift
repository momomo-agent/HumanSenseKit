#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Uses SpeechTranscriber (preferred for live audio) with volatile results.
/// Falls back to DictationTranscriber if SpeechTranscriber is unavailable.
@MainActor
public class SpeechRecognitionManager {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var resultTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var cleanupTask: Task<Void, Never>?
    private var generation: Int = 0

    // nonisolated(unsafe) — written on MainActor, read from audio thread
    nonisolated(unsafe) private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    nonisolated(unsafe) private var analyzerFormat: AVAudioFormat?

    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?
    public var onError: ((Error) -> Void)?

    /// Called from audio render thread — must be nonisolated.
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let continuation = inputContinuation else { return }
        guard let format = analyzerFormat else {
            continuation.yield(AnalyzerInput(buffer: buffer))
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
            continuation.yield(AnalyzerInput(buffer: buffer))
        }
    }
    func startTask() {
        generation += 1
        let myGeneration = generation
        tearDown()

        print("[Speech] startTask gen=\(myGeneration)")

        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: "zh-CN"),
            preset: .progressiveLiveTranscription
        )
        self.transcriber = transcriber

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        // Open immediately — AsyncStream buffers until analyzer starts consuming
        self.inputContinuation = continuation

        // Consume results on background thread — MainActor is busy with ARKit 60fps
        resultTask = Task.detached { [weak self] in
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] Result gen=\(myGeneration): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    await MainActor.run {
                        guard let self, self.generation == myGeneration else { return }
                        self.onResult?(text, isFinal)
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

        // Start analyzer — get format and start ASAP to minimize latency
        let pendingCleanup = cleanupTask
        analyzerTask = Task.detached { [weak self] in
            await pendingCleanup?.value
            let gen = await self?.generation
            guard gen == myGeneration else { return }
            do {
                // Get optimal format and create analyzer in quick succession
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                guard await self?.generation == myGeneration else { return }

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                await MainActor.run {
                    self?.analyzerFormat = format
                    self?.analyzer = analyzer
                }

                // start() returns immediately — analyzer begins consuming stream
                try await analyzer.start(inputSequence: stream)
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
        analyzerTask?.cancel()
        analyzerTask = nil
        inputContinuation = nil
        analyzer = nil
        transcriber = nil
        analyzerFormat = nil

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
                preset: .progressiveLiveTranscription
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
