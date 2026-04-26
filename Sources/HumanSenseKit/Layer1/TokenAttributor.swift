#if os(iOS)
import Foundation

/// Assigns per-token "isFromUser" confidence by querying the correlator's
/// Pearson history at each token's audio time range, then applies median
/// smoothing and gap-bridging to produce clean user-speech runs.
///
/// The output is a list of `AttributedRun`s — contiguous spans of text that
/// belong to the same speaker (user vs other). The demo displays these as
/// the reconstructed "sentence the user said to the screen."
@MainActor
public class TokenAttributor {

    // MARK: - Types

    /// A single token with its attribution confidence.
    public struct AttributedToken {
        public let text: String
        public let audioStart: Double   // seconds relative to audio stream
        public let audioEnd: Double
        public let rawConfidence: Float  // Pearson avg over token's time range
        public let smoothedConfidence: Float  // after median filter
        public let isUser: Bool         // smoothed >= threshold
    }

    /// A contiguous run of tokens from the same speaker.
    public struct AttributedRun: Identifiable {
        public let id = UUID()
        public let text: String
        public let isUser: Bool
        public let avgConfidence: Float
        public let tokens: [AttributedToken]
    }

    // MARK: - Configuration

    /// Pearson threshold for "user speaking"
    public var confidenceThreshold: Float = 0.3

    /// Median filter window (must be odd)
    public var medianWindow: Int = 3

    /// Max consecutive non-user tokens to bridge over
    public var maxGap: Int = 2

    // MARK: - Dependencies

    /// Called to query Pearson history. Set by STTManager/HumanStateEngine.
    public var queryPearson: ((_ wallStart: TimeInterval, _ wallEnd: TimeInterval) -> Float)?

    /// Audio stream start time (wall clock). Set when STTManager stamps it.
    public var audioStreamStartTime: Date?

    // MARK: - State

    /// Latest attributed tokens (updated on each onTokens call)
    public private(set) var currentTokens: [AttributedToken] = []

    /// Latest attributed runs (user speech segments)
    public private(set) var currentRuns: [AttributedRun] = []

    /// The reconstructed user sentence (gap-bridged, user tokens only)
    public private(set) var userSentence: String = ""

    public init() {}

    // MARK: - Public API

    /// Process a batch of tokens from SpeechAnalyzer.
    /// Call this from STTManager.onTokens.
    public func process(tokens: [SpeechToken], isFinal: Bool) {
        guard let startTime = audioStreamStartTime,
              let query = queryPearson else {
            // Can't attribute without timing info
            currentTokens = tokens.map {
                AttributedToken(text: $0.text, audioStart: $0.startTime, audioEnd: $0.endTime,
                                rawConfidence: 0, smoothedConfidence: 0, isUser: false)
            }
            currentRuns = [AttributedRun(text: tokens.map(\.text).joined(), isUser: false, avgConfidence: 0, tokens: currentTokens)]
            userSentence = ""
            return
        }

        // Step 1: Raw confidence — query Pearson history for each token's time range
        let rawConfidences: [Float] = tokens.map { token in
            let wallStart = startTime.timeIntervalSince1970 + token.startTime
            let wallEnd = startTime.timeIntervalSince1970 + token.endTime
            // Convert to systemUptime: we need the offset between Date and systemUptime
            // systemUptime ≈ ProcessInfo.processInfo.systemUptime
            // Date.timeIntervalSince1970 ≈ Date().timeIntervalSince1970
            // offset = Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
            let now = Date()
            let offset = now.timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
            let uptimeStart = wallStart - offset
            let uptimeEnd = wallEnd - offset
            return query(uptimeStart, uptimeEnd)
        }

        // Step 2: Median filter
        let smoothed = medianFilter(rawConfidences, window: medianWindow)

        // Step 3: Build attributed tokens
        currentTokens = zip(tokens, zip(rawConfidences, smoothed)).map { token, confs in
            AttributedToken(
                text: token.text,
                audioStart: token.startTime,
                audioEnd: token.endTime,
                rawConfidence: confs.0,
                smoothedConfidence: confs.1,
                isUser: confs.1 >= confidenceThreshold
            )
        }

        // Step 4: Gap bridge + build runs
        let bridged = gapBridge(currentTokens, maxGap: maxGap)
        currentRuns = buildRuns(from: bridged)

        // Step 5: Extract user sentence
        userSentence = currentRuns
            .filter(\.isUser)
            .map(\.text)
            .joined()
    }

    /// Reset state (e.g. when STT restarts)
    public func reset() {
        currentTokens = []
        currentRuns = []
        userSentence = ""
    }

    // MARK: - Private: Median Filter

    private func medianFilter(_ values: [Float], window: Int) -> [Float] {
        guard values.count >= window else { return values }
        let half = window / 2
        return values.indices.map { i in
            let lo = max(0, i - half)
            let hi = min(values.count - 1, i + half)
            var slice = Array(values[lo...hi])
            slice.sort()
            return slice[slice.count / 2]
        }
    }

    // MARK: - Private: Gap Bridge

    /// If a short run of non-user tokens (≤ maxGap) is surrounded by user
    /// tokens on both sides, flip them to user (bridge the gap).
    private func gapBridge(_ tokens: [AttributedToken], maxGap: Int) -> [AttributedToken] {
        guard tokens.count > 2 else { return tokens }
        var result = tokens

        var i = 0
        while i < result.count {
            if result[i].isUser {
                i += 1
                continue
            }
            // Found a non-user token. Count the gap length.
            var gapEnd = i
            while gapEnd < result.count && !result[gapEnd].isUser {
                gapEnd += 1
            }
            let gapLen = gapEnd - i
            // Bridge if gap is short AND there's a user token on both sides
            if gapLen <= maxGap && i > 0 && result[i - 1].isUser && gapEnd < result.count && result[gapEnd].isUser {
                for j in i..<gapEnd {
                    let t = result[j]
                    result[j] = AttributedToken(
                        text: t.text, audioStart: t.audioStart, audioEnd: t.audioEnd,
                        rawConfidence: t.rawConfidence, smoothedConfidence: t.smoothedConfidence,
                        isUser: true  // bridged
                    )
                }
            }
            i = gapEnd
        }
        return result
    }

    // MARK: - Private: Build Runs

    private func buildRuns(from tokens: [AttributedToken]) -> [AttributedRun] {
        guard !tokens.isEmpty else { return [] }
        var runs: [AttributedRun] = []
        var currentIsUser = tokens[0].isUser
        var currentTokens: [AttributedToken] = [tokens[0]]

        for token in tokens.dropFirst() {
            if token.isUser == currentIsUser {
                currentTokens.append(token)
            } else {
                runs.append(makeRun(tokens: currentTokens, isUser: currentIsUser))
                currentIsUser = token.isUser
                currentTokens = [token]
            }
        }
        runs.append(makeRun(tokens: currentTokens, isUser: currentIsUser))
        return runs
    }

    private func makeRun(tokens: [AttributedToken], isUser: Bool) -> AttributedRun {
        let text = tokens.map(\.text).joined()
        let avg = tokens.isEmpty ? 0 : tokens.map(\.smoothedConfidence).reduce(0, +) / Float(tokens.count)
        return AttributedRun(text: text, isUser: isUser, avgConfidence: avg, tokens: tokens)
    }
}
#endif
