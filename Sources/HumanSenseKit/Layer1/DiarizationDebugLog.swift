#if os(iOS)
import Foundation

/// Thread-safe file logger for speaker diarization debugging.
/// Writes to Documents/diarization-debug.log so we can extract it via
/// `xcrun devicectl device copy from` for post-mortem analysis.
enum DiarizationDebugLog {
    private static let lock = NSLock()
    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static var logURL: URL? {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        return docs.appendingPathComponent("diarization-debug.log")
    }

    static func write(_ message: String) {
        guard let url = logURL else { return }
        let line = "[\(formatter.string(from: Date()))] \(message)\n"
        guard let data = line.data(using: .utf8) else { return }

        lock.lock()
        defer { lock.unlock() }

        if !FileManager.default.fileExists(atPath: url.path) {
            try? data.write(to: url)
            return
        }
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        }
    }
}
#endif
