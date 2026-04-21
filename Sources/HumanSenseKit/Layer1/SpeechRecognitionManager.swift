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
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var resultTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var analyzerFormat: AVAudioFormat?

    /// Monotonically increasing generation counter.
    private var generation: Int = 0

    /// Called with transcription text. `isFinal` means Apple has finalized this segment.
    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?

    /// Called when an error occurs.
    public var onError: ((Error) -> Void)?

    /// Append audio buffer to the analyzer.
    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let format = analyzerFormat else {
            inputContinuation?.yield(AnalyzerInput(buffer: buffer))
            return
        }
        if buffer.format == format {
            inputContinuation?.yield(AnalyzerInput(buffer: buffer))
        } else if let converter = AVAudioConverter(from: buffer.format, to: format) {
            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate
            )
            guard let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
            do {
                try converter.convert(to: converted, from: buffer)
                inputContinuation?.yield(AnalyzerInput(buffer: converted))
            } catch {
                inputContinuation?.yield(AnalyzerInput(buffer: buffer))
            }
        } else {
            inputContinuation?.yield(AnalyzerInput(buffer: buffer))
        }
    }

    /// Start the analyzer with a DictationTranscriber.
    /// Safe to call multiple times — previous analyzer is torn down first.
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

        // Consume results — must use @MainActor Task to access self properties safely
        let onResult = self.onResult
        let onError = self.onError
        resultTask = Task { @MainActor [weak self] in
            do {
                for try await result in transcriber.results {
                    guard let self, self.generation == myGeneration else { return }
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] Result gen=\(myGeneration): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    self.onResult?(text, isFinal)
                }
            } catch {
                guard !Task.isCancelled else { return }
                guard let self, self.generation == myGeneration else { return }
                print("[Speech] Error gen=\(myGeneration): \(error.localizedDescription)")
                self.onError?(error)
            }
        }

        // Start analyzer
        analyzerTask = Task { @MainActor [weak self] in
            do {
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                guard let self, self.generation == myGeneration else { return }
                self.analyzerFormat = format

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                guard self.generation == myGeneration else { return }
                self.analyzer = analyzer

                try await analyzer.start(inputSequence: stream)
                print("[Speech] Analyzer started gen=\(myGeneration)")
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] Analyzer start failed gen=\(myGeneration): \(error.localizedDescription)")
                guard let self, self.generation == myGeneration else { return }
                self.onError?(error)
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

        if let oldContinuation {
            oldContinuation.finish()
        }
        if let oldAnalyzer {
            Task.detached {
                try? await oldAnalyzer.finalizeAndFinishThroughEndOfInput()
            }
        }
    }

    func stop() {
        generation += 1
        tearDown()
    }

    /// Check model availability and download if needed.
    func authorize(completion: @escaping @Sendable (Bool) -> Void) {
        Task { @MainActor in
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
