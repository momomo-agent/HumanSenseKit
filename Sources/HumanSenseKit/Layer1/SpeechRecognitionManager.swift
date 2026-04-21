#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
///
/// Modeled after Apple's official sample code (WWDC25 session 277).
/// NOT @MainActor — SpeechAnalyzer is an actor with its own executor,
/// and forcing MainActor causes dispatch_assert_queue failures.
///
/// Thread safety for appendBuffer (called from audio thread) is handled
/// via the thread-safe AsyncStream.Continuation.yield().
public final class SpeechRecognitionManager: @unchecked Sendable {
    // These are only mutated from startTask/stop which are always called
    // from MainActor (STTManager). appendBuffer only reads analyzerFormat
    // and inputContinuation, which are set before audio starts flowing.
    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var resultTask: Task<Void, Error>?
    private var analyzerFormat: AVAudioFormat?
    private var bufferConverter = BufferConverter()

    /// Called with transcription text on MainActor.
    public var onResult: (@MainActor (_ text: String, _ isFinal: Bool) -> Void)?

    /// Called when an error occurs on MainActor.
    public var onError: (@MainActor (Error) -> Void)?

    /// Append audio buffer to the analyzer.
    /// Called from audio engine thread — only touches thread-safe continuation.
    func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let analyzerFormat, let inputContinuation else { return }
        do {
            let converted = try bufferConverter.convertBuffer(buffer, to: analyzerFormat)
            inputContinuation.yield(AnalyzerInput(buffer: converted))
        } catch {
            inputContinuation.yield(AnalyzerInput(buffer: buffer))
        }
    }

    /// Start the analyzer. Safe to call multiple times.
    /// Must be called from MainActor context (via STTManager).
    func startTask() async {
        await stop()

        print("[Speech] startTask")

        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: "zh-CN"),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: []
        )
        self.transcriber = transcriber

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer

        self.analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputContinuation = continuation

        // Capture callbacks before Task to avoid capturing self
        let onResult = self.onResult
        let onError = self.onError

        // Start result consumption BEFORE analyzer.start() (Apple pattern)
        resultTask = Task.detached {
            do {
                for try await result in transcriber.results {
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    if let onResult {
                        await onResult(text, isFinal)
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                print("[Speech] Error: \(error.localizedDescription)")
                if let onError {
                    await onError(error)
                }
            }
        }

        // Start analyzer — this runs on SpeechAnalyzer's own executor
        do {
            try await analyzer.start(inputSequence: stream)
            print("[Speech] Analyzer started")
        } catch {
            print("[Speech] Analyzer start failed: \(error.localizedDescription)")
            if let onError {
                await onError(error)
            }
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
        Task.detached {
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

private final class BufferConverter: @unchecked Sendable {
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
        let state = ConversionState()

        let status = converter.convert(to: conversionBuffer, error: &nsError) { _, inputStatusPointer in
            if state.processed {
                inputStatusPointer.pointee = .noDataNow
                return nil
            }
            state.processed = true
            inputStatusPointer.pointee = .haveData
            return buffer
        }

        guard status != .error else { throw ConversionError.conversionFailed(nsError) }
        return conversionBuffer
    }

    private class ConversionState {
        var processed = false
    }

    enum ConversionError: Error {
        case failedToCreateConverter
        case failedToCreateBuffer
        case conversionFailed(NSError?)
    }
}
#endif
