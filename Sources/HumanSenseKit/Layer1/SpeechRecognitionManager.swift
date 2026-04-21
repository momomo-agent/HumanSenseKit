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
    private var analyzerFormat: AVAudioFormat?

    /// Called with transcription text. `isFinal` means Apple has finalized this segment.
    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?

    /// Called when an error occurs.
    public var onError: ((Error) -> Void)?

    /// Append audio buffer to the analyzer.
    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let format = analyzerFormat else {
            // If no format conversion needed, pass directly
            inputContinuation?.yield(AnalyzerInput(buffer: buffer))
            return
        }
        // Convert if needed
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
                // Fallback: pass original
                inputContinuation?.yield(AnalyzerInput(buffer: buffer))
            }
        } else {
            inputContinuation?.yield(AnalyzerInput(buffer: buffer))
        }
    }

    /// Start the analyzer with a DictationTranscriber.
    func startTask() {
        stopTask()

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
                guard let self else { return }
                print("[Speech] Error: \(error.localizedDescription)")
                await MainActor.run {
                    self.onError?(error)
                }
            }
        }

        // Start analyzer
        Task {
            do {
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                self.analyzerFormat = format

                let analyzer = SpeechAnalyzer(
                    inputSequence: stream,
                    modules: [transcriber]
                )
                self.analyzer = analyzer
                try await analyzer.start(inputSequence: stream)
                print("[Speech] Analyzer started")
            } catch {
                print("[Speech] Analyzer start failed: \(error.localizedDescription)")
                self.onError?(error)
            }
        }
    }

    /// Stop the analyzer.
    func stopTask() {
        resultTask?.cancel()
        resultTask = nil
        inputContinuation?.finish()
        inputContinuation = nil

        Task {
            try? await analyzer?.finalizeAndFinishThroughEndOfInput()
        }
        analyzer = nil
        transcriber = nil
        analyzerFormat = nil
    }

    /// Finalize current input and restart (for sentence boundary).
    func splitTask() {
        // With SpeechAnalyzer, we don't need manual splitting.
        // Apple handles volatile→final transitions automatically.
        // But if we want to force a boundary, finalize and restart.
        print("[Speech] splitTask — finalizing and restarting")
        let oldContinuation = inputContinuation
        let oldAnalyzer = analyzer

        Task {
            oldContinuation?.finish()
            try? await oldAnalyzer?.finalizeAndFinishThroughEndOfInput()
        }

        // Restart fresh
        startTask()
    }

    func stop() {
        stopTask()
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

            // Check if model is installed
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
