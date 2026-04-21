#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Uses DictationTranscriber with progressiveLongDictation preset for
/// real-time volatile + final results.
@MainActor
public class SpeechRecognitionManager {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: DictationTranscriber?
    private var resultTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var cleanupTask: Task<Void, Never>?
    private var generation: Int = 0

    // nonisolated(unsafe) — accessed from audio thread via appendBuffer
    nonisolated(unsafe) private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    nonisolated(unsafe) private var analyzerFormat: AVAudioFormat?

    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?
    public var onError: ((Error) -> Void)?

    /// Append audio buffer to the analyzer.
    /// Called from the audio render thread — must be nonisolated.
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let continuation = inputContinuation else { return }
        guard let format = analyzerFormat else {
            continuation.yield(AnalyzerInput(buffer: buffer))
            return
        }
        if buffer.format == format {
            continuation.yield(AnalyzerInput(buffer: buffer))
        } else if let converter = AVAudioConverter(from: buffer.format, to: format) {
            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate
            )
            guard let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
            do {
                try converter.convert(to: converted, from: buffer)
                continuation.yield(AnalyzerInput(buffer: converted))
            } catch {
                continuation.yield(AnalyzerInput(buffer: buffer))
            }
        } else {
            continuation.yield(AnalyzerInput(buffer: buffer))
        }
    }
    /// Start the analyzer with a DictationTranscriber.
    func startTask() {
        generation += 1
        let myGeneration = generation
        tearDown()

        print("[Speech] startTask gen=\(myGeneration)")

        let transcriber = DictationTranscriber(
            locale: Locale(identifier: "zh-CN"),
            preset: .progressiveLongDictation
        )
        self.transcriber = transcriber

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = continuation

        // Consume results — detached so Speech framework runs on its own queue
        resultTask = Task.detached { [weak self] in
            do {
                for try await result in transcriber.results {
                    let gen = await self?.generation
                    guard gen == myGeneration else { return }
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] Result gen=\(myGeneration): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    await MainActor.run {
                        self?.onResult?(text, isFinal)
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                let gen = await self?.generation
                guard gen == myGeneration else { return }
                print("[Speech] Error gen=\(myGeneration): \(error.localizedDescription)")
                await MainActor.run {
                    self?.onError?(error)
                }
            }
        }

        // Start analyzer — detached, awaits old cleanup first
        let pendingCleanup = cleanupTask
        analyzerTask = Task.detached { [weak self] in
            await pendingCleanup?.value
            let gen = await self?.generation
            guard gen == myGeneration else { return }
            do {
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                let gen2 = await self?.generation
                guard gen2 == myGeneration else { return }
                await MainActor.run { self?.analyzerFormat = format }

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                let gen3 = await self?.generation
                guard gen3 == myGeneration else { return }
                await MainActor.run { self?.analyzer = analyzer }

                try await analyzer.start(inputSequence: stream)
                print("[Speech] Analyzer started gen=\(myGeneration)")
            } catch {
                guard !Task.isCancelled else { return }
                let gen4 = await self?.generation
                guard gen4 == myGeneration else { return }
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
        cleanupTask = Task.detached {
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
            let supported = await DictationTranscriber.supportedLocales
            guard supported.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) else {
                print("[Speech] zh-CN not supported")
                completion(false)
                return
            }

            let installed = await DictationTranscriber.installedLocales
            if !installed.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
                print("[Speech] zh-CN model not installed, attempting download...")
                let transcriber = DictationTranscriber(locale: locale, preset: .progressiveLongDictation)
                if let downloader = try? await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    do {
                        try await downloader.downloadAndInstall()
                        print("[Speech] Model downloaded")
                    } catch {
                        print("[Speech] Model download failed: \(error.localizedDescription)")
                        completion(false)
                        return
                    }
                }
            }

            print("[Speech] Authorized and ready")
            completion(true)
        }
    }
}
#endif
