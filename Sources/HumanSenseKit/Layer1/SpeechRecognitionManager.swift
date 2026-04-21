#if os(iOS)
import Foundation
import Speech
import AVFoundation

/// Layer 1: Speech recognition using iOS 26 SpeechAnalyzer.
@MainActor
public class SpeechRecognitionManager {
    private var analyzer: SpeechAnalyzer?
    private var transcriber: DictationTranscriber?
    private var resultTask: Task<Void, Never>?
    private var analyzerTask: Task<Void, Never>?
    private var cleanupTask: Task<Void, Never>?
    private var generation: Int = 0

    // nonisolated(unsafe) — written on MainActor, read from audio thread
    nonisolated(unsafe) private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    nonisolated(unsafe) private var analyzerFormat: AVAudioFormat?
    // Gate: only yield buffers after analyzer is running
    nonisolated(unsafe) private var analyzerReady: Bool = false

    public var onResult: ((_ text: String, _ isFinal: Bool) -> Void)?
    public var onError: ((Error) -> Void)?

    /// Called from audio render thread — must be nonisolated.
    nonisolated func appendBuffer(_ buffer: AVAudioPCMBuffer) {
        guard analyzerReady, let continuation = inputContinuation else { return }
        // Copy the buffer — audio tap may reuse the original
        guard let copy = Self.copyBuffer(buffer) else { return }
        guard let format = analyzerFormat else {
            continuation.yield(AnalyzerInput(buffer: copy))
            return
        }
        if copy.format == format {
            continuation.yield(AnalyzerInput(buffer: copy))
        } else if let converter = AVAudioConverter(from: copy.format, to: format),
                  let converted = Self.convert(copy, to: format, using: converter) {
            continuation.yield(AnalyzerInput(buffer: converted))
        } else {
            continuation.yield(AnalyzerInput(buffer: copy))
        }
    }

    private nonisolated static func copyBuffer(_ src: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(pcmFormat: src.format, frameCapacity: src.frameLength) else { return nil }
        copy.frameLength = src.frameLength
        if let srcData = src.floatChannelData, let dstData = copy.floatChannelData {
            for ch in 0..<Int(src.format.channelCount) {
                memcpy(dstData[ch], srcData[ch], Int(src.frameLength) * MemoryLayout<Float>.size)
            }
        } else if let srcData = src.int16ChannelData, let dstData = copy.int16ChannelData {
            for ch in 0..<Int(src.format.channelCount) {
                memcpy(dstData[ch], srcData[ch], Int(src.frameLength) * MemoryLayout<Int16>.size)
            }
        }
        return copy
    }

    private nonisolated static func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat, using converter: AVAudioConverter) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate)
        guard let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return nil }
        try? converter.convert(to: converted, from: buffer)
        return converted
    }
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
        // Don't set inputContinuation yet — wait until analyzer is ready
        // This prevents buffering audio before the analyzer can consume it

        // Consume results
        resultTask = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    guard let self, self.generation == myGeneration else { return }
                    let text = String(result.text.characters)
                    let isFinal = result.isFinal
                    print("[Speech] Result gen=\(myGeneration): '\(text.prefix(60))' isFinal=\(isFinal ? 1 : 0)")
                    self.onResult?(text, isFinal)
                }
            } catch {
                guard let self, !Task.isCancelled, self.generation == myGeneration else { return }
                print("[Speech] Error gen=\(myGeneration): \(error.localizedDescription)")
                self.onError?(error)
            }
        }

        // Start analyzer — await old cleanup, then open the buffer gate
        let pendingCleanup = cleanupTask
        analyzerTask = Task { [weak self] in
            await pendingCleanup?.value
            guard let self, self.generation == myGeneration else { return }
            do {
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
                guard self.generation == myGeneration else { return }
                self.analyzerFormat = format

                let analyzer = SpeechAnalyzer(modules: [transcriber])
                guard self.generation == myGeneration else { return }
                self.analyzer = analyzer

                try await analyzer.start(inputSequence: stream)
                print("[Speech] Analyzer started gen=\(myGeneration)")

                // Open the gate after analyzer is consuming the stream
                self.inputContinuation = continuation
                self.analyzerReady = true
            } catch {
                guard !Task.isCancelled, self.generation == myGeneration else { return }
                print("[Speech] Analyzer start failed gen=\(myGeneration): \(error.localizedDescription)")
                self.onError?(error)
            }
        }
    }

    private func tearDown() {
        analyzerReady = false
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
