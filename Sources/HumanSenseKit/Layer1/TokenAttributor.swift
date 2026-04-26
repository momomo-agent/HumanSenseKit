#if os(iOS)
import Foundation

/// Assigns per-token "isFromUser" confidence by querying the correlator's
/// history at each token's audio time range, then applies:
///  1. Multi-signal confidence (Pearson + jaw peak rate + jaw std)
///  2. Median smoothing to kill spike noise
///  3. Gap bridging to ignore 1-2 token dropouts
///  4. **Sentence-level majority vote** — if the sentence is dominantly
///     the user, fill in the edges and middle dips (boundary & gap recovery)
///
/// The output is the reconstructed `userSentence`: what we think the user
/// actually said to the screen, even when a few characters fell below the
/// instantaneous Pearson threshold (cold start, side profile, brief head turn).
@MainActor
public class TokenAttributor {

    // MARK: - Types

    /// A single token with its attribution confidence + feature bundle.
    public struct AttributedToken {
        public let text: String
        public let audioStart: Double
        public let audioEnd: Double
        public let rawConfidence: Float        // base Pearson
        public let fusedConfidence: Float      // Pearson + jaw signals fused
        public let smoothedConfidence: Float   // median filter
        public let isUser: Bool                // final verdict
        public let filledBySentence: Bool      // upgraded by sentence vote
        public let features: LipAudioCorrelator.TokenFeatures
    }

    /// A contiguous run of tokens from the same speaker.
    public struct AttributedRun: Identifiable {
        public let id = UUID()
        public let text: String
        public let isUser: Bool
        public let avgConfidence: Float
        public let tokens: [AttributedToken]
    }

    /// Sentence-level aggregate verdict.
    public struct SentenceVerdict {
        public let userTokenRatio: Float        // count(isUser) / total
        public let peakConfidence: Float        // max smoothedConfidence
        public let longestUserSpan: Float       // longest contiguous user run / total
        public let isUserDominant: Bool
    }

    // MARK: - Configuration

    public var confidenceThreshold: Float = 0.3
    public var medianWindow: Int = 3
    public var maxGap: Int = 2

    /// Sentence vote thresholds — if any two trigger, the sentence is "yours"
    public var sentenceUserRatioThreshold: Float = 0.5
    public var sentencePeakThreshold: Float = 0.45
    public var sentenceSpanThreshold: Float = 0.35

    // MARK: - Dependencies

    /// Closure to query rich features from the correlator.
    public var queryFeatures: ((_ wallStart: TimeInterval, _ wallEnd: TimeInterval) -> LipAudioCorrelator.TokenFeatures)?

    /// Fallback closure (Pearson-only) for backwards compat.
    public var queryPearson: ((_ wallStart: TimeInterval, _ wallEnd: TimeInterval) -> Float)?

    /// Audio stream start time (wall clock).
    public var audioStreamStartTime: Date?

    // MARK: - State

    public private(set) var currentTokens: [AttributedToken] = []
    public private(set) var currentRuns: [AttributedRun] = []
    public private(set) var userSentence: String = ""
    public private(set) var lastVerdict: SentenceVerdict = .init(
        userTokenRatio: 0, peakConfidence: 0, longestUserSpan: 0, isUserDominant: false
    )

    public init() {}

    // MARK: - Public API

    public func process(tokens: [SpeechToken], isFinal: Bool) {
        guard let startTime = audioStreamStartTime else {
            fallbackEmpty(tokens: tokens)
            return
        }

        // Convert Date → systemUptime offset
        let now = Date()
        let offset = now.timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime

        // Step 1: gather per-token features
        let tokenFeatures: [LipAudioCorrelator.TokenFeatures] = tokens.map { token in
            let wallStart = startTime.timeIntervalSince1970 + token.startTime
            let wallEnd = startTime.timeIntervalSince1970 + token.endTime
            let uptimeStart = wallStart - offset
            let uptimeEnd = wallEnd - offset
            if let qf = queryFeatures {
                return qf(uptimeStart, uptimeEnd)
            }
            if let qp = queryPearson {
                let p = qp(uptimeStart, uptimeEnd)
                return LipAudioCorrelator.TokenFeatures(
                    avgPearson: p, maxPearson: p, jawStd: 0, jawPeakRate: 0,
                    faceVisibleRatio: 1, sampleCount: 0
                )
            }
            return LipAudioCorrelator.TokenFeatures(
                avgPearson: 0, maxPearson: 0, jawStd: 0, jawPeakRate: 0,
                faceVisibleRatio: 0, sampleCount: 0
            )
        }

        // Step 2: fuse signals into confidence
        let fused = tokenFeatures.map { fuseConfidence($0) }
        let raw = tokenFeatures.map { $0.avgPearson }

        // Step 3: median smooth
        let smoothed = medianFilter(fused, window: medianWindow)

        // Step 4: initial per-token verdict
        var isUserArr = smoothed.map { $0 >= confidenceThreshold }

        // Step 5: gap bridge (short non-user runs between user tokens)
        isUserArr = gapBridge(isUserArr, maxGap: maxGap)

        // Step 6: sentence-level majority vote
        let verdict = computeVerdict(isUserArr: isUserArr, smoothed: smoothed)
        lastVerdict = verdict

        var filledBySentence = Array(repeating: false, count: tokens.count)
        if verdict.isUserDominant {
            // Fill in the gaps — all tokens in the sentence become user.
            // This recovers cold-start head, tail droop, and mid-sentence dips.
            for i in isUserArr.indices where !isUserArr[i] {
                // Don't upgrade tokens with clearly non-user signal
                // (very low raw + low fused + face not visible at all)
                let f = tokenFeatures[i]
                let clearlyNotUser = raw[i] < -0.1 && f.faceVisibleRatio < 0.2
                if !clearlyNotUser {
                    isUserArr[i] = true
                    filledBySentence[i] = true
                }
            }
        }

        // Step 7: build AttributedTokens
        currentTokens = tokens.indices.map { i in
            AttributedToken(
                text: tokens[i].text,
                audioStart: tokens[i].startTime,
                audioEnd: tokens[i].endTime,
                rawConfidence: raw[i],
                fusedConfidence: fused[i],
                smoothedConfidence: smoothed[i],
                isUser: isUserArr[i],
                filledBySentence: filledBySentence[i],
                features: tokenFeatures[i]
            )
        }

        // Step 8: build runs + reconstruct userSentence
        currentRuns = buildRuns(from: currentTokens)
        userSentence = currentRuns.filter(\.isUser).map(\.text).joined()
    }

    public func reset() {
        currentTokens = []
        currentRuns = []
        userSentence = ""
        lastVerdict = .init(userTokenRatio: 0, peakConfidence: 0, longestUserSpan: 0, isUserDominant: false)
    }

    // MARK: - Signal fusion

    /// Combine Pearson + jaw peak rate + jaw std into a single confidence.
    /// The Pearson is the main signal; jaw signals act as:
    ///  - boost during Pearson cold-start (first ~500ms of speaking)
    ///  - penalty when audio is active but mouth isn't moving (someone else speaking)
    private func fuseConfidence(_ f: LipAudioCorrelator.TokenFeatures) -> Float {
        var score = f.avgPearson

        // Boost if jaw peak rate is in the speech band (3-8 Hz)
        if f.jawPeakRate >= 3 && f.jawPeakRate <= 8 {
            score += 0.15
        }

        // Boost from max Pearson (catches brief strong alignment even if avg is noisy)
        if f.maxPearson > 0.5 {
            score += 0.05 * min(1, (f.maxPearson - 0.5) / 0.3)
        }

        // Penalty: jaw barely moving but we have a token (someone else is speaking)
        if f.jawStd < 0.005 && f.faceVisibleRatio > 0.5 {
            score -= 0.3
        }

        // If face wasn't visible, don't over-penalize — leave confidence ambiguous
        // so the sentence-level vote can decide.
        if f.faceVisibleRatio < 0.3 {
            score = max(score, 0)  // floor at 0, don't let penalties drive it negative
        }

        return max(-1, min(1, score))
    }

    // MARK: - Verdict

    private func computeVerdict(isUserArr: [Bool], smoothed: [Float]) -> SentenceVerdict {
        guard !isUserArr.isEmpty else {
            return .init(userTokenRatio: 0, peakConfidence: 0, longestUserSpan: 0, isUserDominant: false)
        }
        let userCount = isUserArr.filter { $0 }.count
        let ratio = Float(userCount) / Float(isUserArr.count)
        let peak = smoothed.max() ?? 0

        // longest contiguous user run
        var longest = 0, cur = 0
        for b in isUserArr {
            if b { cur += 1; longest = max(longest, cur) }
            else { cur = 0 }
        }
        let spanRatio = Float(longest) / Float(isUserArr.count)

        // Trigger any two
        let ratioHit = ratio >= sentenceUserRatioThreshold
        let peakHit = peak >= sentencePeakThreshold
        let spanHit = spanRatio >= sentenceSpanThreshold
        let hits = [ratioHit, peakHit, spanHit].filter { $0 }.count
        let dominant = hits >= 2

        return .init(
            userTokenRatio: ratio,
            peakConfidence: peak,
            longestUserSpan: spanRatio,
            isUserDominant: dominant
        )
    }

    // MARK: - Median filter

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

    // MARK: - Gap bridge

    private func gapBridge(_ isUser: [Bool], maxGap: Int) -> [Bool] {
        guard isUser.count > 2 else { return isUser }
        var result = isUser
        var i = 0
        while i < result.count {
            if result[i] { i += 1; continue }
            var gapEnd = i
            while gapEnd < result.count && !result[gapEnd] { gapEnd += 1 }
            let gapLen = gapEnd - i
            if gapLen <= maxGap && i > 0 && result[i - 1] && gapEnd < result.count && result[gapEnd] {
                for j in i..<gapEnd { result[j] = true }
            }
            i = gapEnd
        }
        return result
    }

    // MARK: - Build runs

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

    // MARK: - Fallbacks

    private func fallbackEmpty(tokens: [SpeechToken]) {
        let empty = LipAudioCorrelator.TokenFeatures(
            avgPearson: 0, maxPearson: 0, jawStd: 0, jawPeakRate: 0, faceVisibleRatio: 0, sampleCount: 0
        )
        currentTokens = tokens.map {
            AttributedToken(
                text: $0.text, audioStart: $0.startTime, audioEnd: $0.endTime,
                rawConfidence: 0, fusedConfidence: 0, smoothedConfidence: 0,
                isUser: false, filledBySentence: false, features: empty
            )
        }
        currentRuns = [AttributedRun(text: tokens.map(\.text).joined(), isUser: false, avgConfidence: 0, tokens: currentTokens)]
        userSentence = ""
    }
}
#endif
