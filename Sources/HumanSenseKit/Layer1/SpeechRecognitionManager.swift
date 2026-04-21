#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Uses DictationTranscriber with progressiveLongDictation preset for
/// real-time volatile + final results. No more manual task splitting —
/// Apple handles the volatile→final lifecycle.
@MainActor
public class SpeechRecognitionManager {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: DictationTranscriber?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var resultTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var analyzerFormat: AVAudioFormat?
    private var isStarting = false

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
    func startTask() {
        guard !isStarting else {
            print("[Speech] startTask skipped — already starting")
            return
        }
        isStarting = true

        // Stop any existing analyzer first (synchronously cancel)
        resultTask?.cancel()
        resultTask = nil
        analyzerTask?.cancel()
        analyzerTask = nil
        inputContinuation?.finish()
        inputContinuation = nil
        analyzer = nil
        transcriber = nil
        analyzerFormat = nil

        let transcriber = DictationTranscriber(
            locale: Locale(identifier: "zh-CN"),
            preset: .progressiveLongDictation
        )
        self.transcriber = transcriber

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = continuation

        // Start result consumption
        resultTask = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    guard let self else { return }
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] Result: '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    await MainActor.run {
                        self.onResult?(text, isFinal)
                    }
                }
            } catch {
                guard let self, !Task.isCancelled else { return }
                print("[Speech] Error: \(error.localizedDescription)")
                await MainActor.run {
                    self.onError?(error)
                }
            }
        }

        // Start analyzer — only use start(), not the inputSequence init
        analyzerTask = Task { [weak self] in
            do {
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                await MainActor.run { self?.analyzerFormat = format }

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                await MainActor.run {
                    self?.analyzer = analyzer
                    self?.isStarting = false
                }
                try await analyzer.start(inputSequence: stream)
                print("[Speech] Analyzer started")
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] Analyzer start failed: \(error.localizedDescription)")
                await MainActor.run {
                    self?.isStarting = false
                    self?.onError?(error)
                }
            }
        }
    }

    func stop() {
        resultTask?.cancel()
        resultTask = nil
        analyzerTask?.cancel()
        analyzerTask = nil
        inputContinuation?.finish()
        inputContinuation = nil
        analyzer = nil
        transcriber = nil
        analyzerFormat = nil
        isStarting = false
    }

    /// Check model availability and download if needed.
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
