#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Based on Apple's official sample code "Bringing advanced speech-to-text
/// capabilities to your app" (WWDC25 session 277).
///
/// Key design decisions from Apple's sample:
/// - Uses SpeechTranscriber (not DictationTranscriber) for live audio
/// - NOT @MainActor — uses @Observable pattern instead
/// - Result consumption task starts BEFORE analyzer.start()
/// - Buffer conversion uses a reusable BufferConverter
/// - finishTranscribing() calls finalizeAndFinishThroughEndOfInput()
public class SpeechRecognitionManager {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var resultTask: Task<Void, Error>?
    private var analyzerFormat: AVAudioFormat?
    private var bufferConverter = BufferConverter()

    /// Called with transcription text. `isFinal` means Apple has finalized this segment.
    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?

    /// Called when an error occurs.
    public var onError: ((Error) -> Void)?

    /// Append audio buffer to the analyzer.
    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let analyzerFormat, let inputContinuation else { return }
        do {
            let converted = try bufferConverter.convertBuffer(buffer, to: analyzerFormat)
            inputContinuation.yield(AnalyzerInput(buffer: converted))
        } catch {
            // Fallback: try raw buffer
            inputContinuation.yield(AnalyzerInput(buffer: buffer))
        }
    }

    /// Start the analyzer. Safe to call multiple times — previous session is torn down first.
    func startTask() async {
        // Tear down any existing session
        await stop()

        print("[Speech] startTask")

        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: "zh-CN"),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: []
        )
        self.transcriber = transcriber

        // Create analyzer with modules (Apple pattern: init with modules, start with inputSequence)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer

        // Get best audio format
        self.analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        // Create input stream
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = continuation

        // Start result consumption BEFORE analyzer.start() (Apple pattern)
        let onResult = self.onResult
        let onError = self.onError
        resultTask = Task {
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    await MainActor.run {
                        onResult?(text, isFinal)
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] Error: \(error.localizedDescription)")
                await MainActor.run {
                    onError?(error)
                }
            }
        }

        // Start analyzer with input sequence
        do {
            try await analyzer.start(inputSequence: stream)
            print("[Speech] Analyzer started")
        } catch {
            print("[Speech] Analyzer start failed: \(error.localizedDescription)")
            onError?(error)
        }
    }

    /// Stop and tear down.
    func stop() async {
        inputContinuation?.finish()
        inputContinuation = nil

        if let analyzer {
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
        }

        resultTask?.cancel()
        resultTask = nil
        analyzer = nil
        transcriber = nil
        analyzerFormat = nil
    }

    /// Check model availability and download if needed.
    func authorize(completion: @escaping @Sendable (Bool) -> Void) {
        Task {
            let locale = Locale(identifier: "zh-CN")
            let supported = await SpeechTranscriber.supportedLocales
            guard supported.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) else {
                print("[Speech] zh-CN not supported")
                completion(false)
                return
            }

            let installed = await SpeechTranscriber.installedLocales
            if !installed.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
                print("[Speech] zh-CN model not installed, attempting download...")
                let transcriber = SpeechTranscriber(
                    locale: locale,
                    transcriptionOptions: [],
                    reportingOptions: [.volatileResults],
                    attributeOptions: []
                )
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

// MARK: - Buffer Converter (from Apple sample code)

/// Reusable audio buffer converter that handles format mismatches.
private class BufferConverter {
    private var converter: AVAudioConverter?

    func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format
        guard inputFormat != format else { return buffer }

        if converter == nil || converter?.outputFormat != format {
            converter = AVAudioConverter(from: inputFormat, to: format)
            converter?.primeMethod = .none
        }

        guard let converter else { throw ConversionError.failedToCreateConverter }

        let sampleRateRatio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let scaledLength = Double(buffer.frameLength) * sampleRateRatio
        let frameCapacity = AVAudioFrameCount(scaledLength.rounded(.up))
        guard let conversionBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: frameCapacity) else {
            throw ConversionError.failedToCreateBuffer
        }

        var nsError: NSError?
        var bufferProcessed = false

        let status = converter.convert(to: conversionBuffer, error: &nsError) { _, inputStatusPointer in
            defer { bufferProcessed = true }
            inputStatusPointer.pointee = bufferProcessed ? .noDataNow : .haveData
            return bufferProcessed ? nil : buffer
        }

        guard status != .error else { throw ConversionError.conversionFailed(nsError) }
        return conversionBuffer
    }

    enum ConversionError: Error {
        case failedToCreateConverter
        case failedToCreateBuffer
        case conversionFailed(NSError?)
    }
}
#endif
