//
//  UserSentenceReconstructor.swift
//  HumanSenseKit
//
//  Attributes STT tokens to the user by fusing per-token audio/face signals,
//  then rebuilds the live/just-finalized user sentence with gap-rescue logic.
//
//  Design summary:
//   * Per-token decision (`decideIsUser`) uses a 3-layer rule stack:
//       L1 Gate — must be looking at screen & head forward
//       L2 Main — jaw activity (maxJaw / jawStd) as primary signal
//       L3 Pearson — filter/rescue using jaw-vs-vol correlation
//   * Sentence-level vote (`applySentenceVote`) upgrades low-confidence rows
//     inside an otherwise user-dominant sentence, but REQUIRES jaw movement
//     (gaze+head-forward alone aren't enough — that only proves the user was
//     watching, not speaking).
//   * Final sentence reconstruction (`reconstructSentence`) scans only the
//     most recent final batch (no accumulation across past utterances) and
//     uses a per-sentence budget to absorb small misclassification gaps.
//     Cost per gap = 3 - presenceHits; mouth-moved is mandatory.
//
//  This reconstructor supersedes TokenAttributor + SentenceBuilder for apps
//  that want a single reconstructed "what the user just said" string. The
//  older machinery is still present for backwards compatibility but callers
//  should migrate to `HumanStateEngine.userSentenceReconstructor`.
//

#if os(iOS)
import Foundation

@MainActor
public final class UserSentenceReconstructor: ObservableObject {

    // MARK: - Types

    public struct Sample: Sendable {
        public let ts: Double
        public let jaw: Float
        public let vol: Float
        public let gaze: Bool
        public let headFwd: Bool

        public init(ts: Double, jaw: Float, vol: Float, gaze: Bool, headFwd: Bool) {
            self.ts = ts
            self.jaw = jaw
            self.vol = vol
            self.gaze = gaze
            self.headFwd = headFwd
        }
    }

    /// One row of token-aligned measurements + the verdict.
    public struct TokenRow: Identifiable, Sendable {
        public let id = UUID()
        public let text: String
        public let startTime: Double
        public let endTime: Double
        public let avgJaw: Float
        public let maxJaw: Float
        public let jawStd: Float
        public let avgVol: Float
        public let maxVol: Float
        public let volStd: Float
        public let localPearson: Float
        public let fusedScore: Float
        public let gazeRatio: Float
        public let headFwdRatio: Float
        public let sampleCount: Int
        public let effectiveWindow: Double
        public let isUser: Bool
        public let filledBySentence: Bool
        public let isFinal: Bool
    }

    public struct Verdict: Sendable {
        public let userRatio: Float
        public let peakScore: Float
        public let longestSpanRatio: Float
        public let isUserDominant: Bool

        public static let zero = Verdict(
            userRatio: 0, peakScore: 0, longestSpanRatio: 0, isUserDominant: false
        )
    }

    // MARK: - Default thresholds (static for UI use)

    public static let defaultJawActivityThreshold: Float = 0.12
    public static let defaultJawStdThreshold: Float = 0.008
    public static let defaultVolActiveThreshold: Float = 0.008
    public static let defaultGazeRatioThreshold: Float = 0.2
    public static let defaultHeadFwdRatioThreshold: Float = 0.2
    public static let defaultGapBudgetPerSentence: Int = 2

    // Back-compat aliases so existing debug UIs keep compiling.
    public static var gazeRatioThreshold: Float { defaultGazeRatioThreshold }
    public static var headFwdRatioThreshold: Float { defaultHeadFwdRatioThreshold }

    // MARK: - Tunable thresholds (public so apps can tune).
    // Defaults mirror the `default*` statics above; keep them in sync when
    // tuning. We can't reference `Self.default...` in stored-property
    // initializers on a non-final class, so literals are repeated here.

    /// maxJaw in window ≥ this → probably speaking
    public var jawActivityThreshold: Float = 0.12
    /// jaw std ≥ this → mouth moving
    public var jawStdThreshold: Float = 0.008
    /// vol present
    public var volActiveThreshold: Float = 0.008
    /// below this sample count we expand the window
    public var minSampleCount: Int = 3
    /// ±80 ms expansion per try
    public var windowExpandMs: Double = 0.08
    /// cap expansion
    public var maxWindowMs: Double = 0.30
    /// ≥20% of window frames looking at screen ⇒ "looking"
    public var gazeRatioThreshold: Float = 0.2
    public var headFwdRatioThreshold: Float = 0.2
    /// How many gap-budget points a sentence starts with.
    /// Cost per gap = 3 - presenceHits.
    public var gapBudgetPerSentence: Int = 2

    // MARK: - State

    private var samples: [Sample] = []
    private let maxSamples = 6000

    private var finalizedTokens: [TokenRow] = []
    private var volatileTokens: [TokenRow] = []
    /// Just the rows from the MOST RECENT final callback.
    private var lastFinalBatch: [TokenRow] = []

    @Published public private(set) var tokens: [TokenRow] = []
    @Published public private(set) var lastVerdict: Verdict = .zero
    /// Reconstructed user sentence from the current volatile or most recent final batch.
    @Published public private(set) var userSentence: String = ""

    public init() {}

    // MARK: - Input API

    /// Feed one frame of sensor state. Call at ~60 Hz.
    public func recordSample(ts: Double, jaw: Float, vol: Float, gaze: Bool, headFwd: Bool) {
        samples.append(Sample(ts: ts, jaw: jaw, vol: vol, gaze: gaze, headFwd: headFwd))
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
    }

    /// Feed tokens from the STT backend.
    /// - Parameters:
    ///   - incoming: tokens as emitted by SpeechTranscriber (each run has audioTimeRange)
    ///   - isFinal: whether this is a stabilized final result
    ///   - audioStreamStart: the wall-clock time the audio stream started; tokens'
    ///     `startTime`/`endTime` are offsets from this point.
    public func recordTokens(
        _ incoming: [SpeechToken],
        isFinal: Bool,
        audioStreamStart: Date?
    ) {
        let base = audioStreamStart?.timeIntervalSince1970 ?? 0
        let rows = buildRows(from: incoming, base: base, isFinal: isFinal)
        let votedRows = applySentenceVote(rows)

        if isFinal {
            finalizedTokens.append(contentsOf: votedRows)
            lastFinalBatch = votedRows
            volatileTokens.removeAll()
            if finalizedTokens.count > 500 {
                finalizedTokens.removeFirst(finalizedTokens.count - 500)
            }
        } else {
            volatileTokens = votedRows
        }
        tokens = finalizedTokens + volatileTokens

        userSentence = reconstructSentence(volatile: volatileTokens, finalBatch: lastFinalBatch)
    }

    public func clear() {
        finalizedTokens.removeAll()
        volatileTokens.removeAll()
        lastFinalBatch.removeAll()
        tokens.removeAll()
        samples.removeAll()
        userSentence = ""
        lastVerdict = .zero
    }

    // MARK: - Range query API (for legacy adapters, e.g. STTManager.segments override)

    /// Result of querying user-attribution over an audio time range.
    public struct RangeAttribution: Sendable {
        /// Number of tokens whose `[startTime, endTime]` overlaps the queried range.
        public let tokenCount: Int
        /// Number of overlapping tokens classified as user (after sentence vote).
        public let userTokenCount: Int
        /// userTokenCount / tokenCount. 0 when tokenCount == 0.
        public let userRatio: Float
        /// Whether the range had any overlapping tokens at all.
        public var hasCoverage: Bool { tokenCount > 0 }
    }

    /// Attribute an arbitrary audio time range (seconds relative to stream start)
    /// against the tokens this reconstructor has seen.
    ///
    /// Used by `STTManager.rebuildSegments()` to transparently upgrade
    /// `SpeechSegment.isFromUser` without changing the builder itself.
    /// - Parameters:
    ///   - start: audio-stream-relative seconds (nil → hasCoverage=false)
    ///   - end: audio-stream-relative seconds (nil → hasCoverage=false)
    public func attribution(for start: Double?, end: Double?) -> RangeAttribution {
        guard let start, let end, end >= start else {
            return RangeAttribution(tokenCount: 0, userTokenCount: 0, userRatio: 0)
        }
        // Walk both finalized + volatile tokens. Overlap = any intersection.
        var total = 0
        var userCount = 0
        let all = finalizedTokens + volatileTokens
        for row in all {
            // Overlap test (inclusive).
            guard row.endTime >= start && row.startTime <= end else { continue }
            total += 1
            if row.isUser { userCount += 1 }
        }
        let ratio = total > 0 ? Float(userCount) / Float(total) : 0
        return RangeAttribution(tokenCount: total, userTokenCount: userCount, userRatio: ratio)
    }

    /// Returns the concatenated text of tokens whose audio range overlaps
    /// [start, end] AND whose per-token verdict is `isUser == true`.
    ///
    /// Used by `STTManager.rebuildSegments()` to transparently replace
    /// `SpeechSegment.text` with the user-only subset, so existing apps
    /// that render `segment.text` get a stream of "just what the user
    /// said at the screen" without any code change.
    ///
    /// Returns `nil` when there is no coverage (no overlapping tokens at
    /// all). Returns an empty string when there are overlapping tokens but
    /// none are user — callers should treat this as "user said nothing
    /// here" and typically skip rendering.
    public func userTextInRange(start: Double?, end: Double?) -> String? {
        guard let start, let end, end >= start else { return nil }
        var pieces: [String] = []
        var anyOverlap = false
        let all = finalizedTokens + volatileTokens
        for row in all {
            guard row.endTime >= start && row.startTime <= end else { continue }
            anyOverlap = true
            if row.isUser { pieces.append(row.text) }
        }
        guard anyOverlap else { return nil }
        return pieces.joined()
    }

    // MARK: - Row construction

    /// One SpeechToken = one row. Each token carries a precise `audioTimeRange`
    /// per run; splitting the range across characters by equal division destroys
    /// the alignment (front chars end up with n=0 samples). Keep tokens whole —
    /// a row's text may be a single character, a word, or a short phrase.
    private func buildRows(from tokens: [SpeechToken], base: Double, isFinal: Bool) -> [TokenRow] {
        var rows: [TokenRow] = []
        for t in tokens {
            let wallStart = base + t.startTime
            let wallEnd = base + t.endTime
            guard !t.text.isEmpty else { continue }
            rows.append(buildRow(text: t.text, wallStart: wallStart, wallEnd: wallEnd, isFinal: isFinal))
        }
        return rows
    }

    private func buildRow(text: String, wallStart: Double, wallEnd: Double, isFinal: Bool) -> TokenRow {
        var expand: Double = 0
        var matched = samples.filter { $0.ts >= wallStart && $0.ts <= wallEnd }
        while matched.count < minSampleCount && expand < maxWindowMs {
            expand += windowExpandMs
            let s = wallStart - expand
            let e = wallEnd + expand
            matched = samples.filter { $0.ts >= s && $0.ts <= e }
        }

        let jaws = matched.map { $0.jaw }
        let vols = matched.map { $0.vol }
        let n = matched.count

        let jawAvg = n == 0 ? 0 : jaws.reduce(0, +) / Float(n)
        let volAvg = n == 0 ? 0 : vols.reduce(0, +) / Float(n)
        let jawStdVal = std(jaws, mean: jawAvg)
        let volStdVal = std(vols, mean: volAvg)

        let pearson = localPearson(jaws: jaws, jawMean: jawAvg, vols: vols, volMean: volAvg, jawStd: jawStdVal, volStd: volStdVal)
        let fused = fuse(pearson: pearson, jawStd: jawStdVal, volStd: volStdVal, volAvg: volAvg)

        let gazeCount = matched.filter { $0.gaze }.count
        let headCount = matched.filter { $0.headFwd }.count
        let gazeRatio: Float = n > 0 ? Float(gazeCount) / Float(n) : 0
        let headFwdRatio: Float = n > 0 ? Float(headCount) / Float(n) : 0

        return TokenRow(
            text: text,
            startTime: wallStart,
            endTime: wallEnd,
            avgJaw: jawAvg,
            maxJaw: jaws.max() ?? 0,
            jawStd: jawStdVal,
            avgVol: volAvg,
            maxVol: vols.max() ?? 0,
            volStd: volStdVal,
            localPearson: pearson,
            fusedScore: fused,
            gazeRatio: gazeRatio,
            headFwdRatio: headFwdRatio,
            sampleCount: n,
            effectiveWindow: (wallEnd - wallStart) + 2 * expand,
            isUser: decideIsUser(
                maxJaw: jaws.max() ?? 0, jawStd: jawStdVal, volStd: volStdVal,
                volAvg: volAvg, pearson: pearson,
                gazeRatio: gazeRatio, headFwdRatio: headFwdRatio
            ),
            filledBySentence: false,
            isFinal: isFinal
        )
    }

    // MARK: - Math

    private func std(_ v: [Float], mean: Float) -> Float {
        guard v.count >= 2 else { return 0 }
        let ss = v.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
        return sqrt(ss / Float(v.count))
    }

    private func localPearson(jaws: [Float], jawMean: Float, vols: [Float], volMean: Float, jawStd: Float, volStd: Float) -> Float {
        guard jaws.count == vols.count, jaws.count >= 2 else { return 0 }
        guard jawStd > 0.0005 && volStd > 0.0005 else { return 0 }
        var sum: Float = 0
        for i in jaws.indices {
            sum += (jaws[i] - jawMean) * (vols[i] - volMean)
        }
        let cov = sum / Float(jaws.count)
        let r = cov / (jawStd * volStd)
        return max(-1, min(1, r))
    }

    private func fuse(pearson: Float, jawStd: Float, volStd: Float, volAvg: Float) -> Float {
        var score: Float = 0
        if jawStd >= jawStdThreshold { score += 0.4 }
        if volStd > 0.005 { score += 0.2 }
        if pearson > 0.1 { score += 0.2 }
        if pearson > 0.3 { score += 0.2 }
        if jawStd < 0.004 && volAvg > 0.02 { score -= 0.5 }
        return max(-1, min(1, score))
    }

    // MARK: - Per-token verdict

    private func decideIsUser(
        maxJaw: Float, jawStd: Float, volStd: Float, volAvg: Float,
        pearson: Float, gazeRatio: Float, headFwdRatio: Float
    ) -> Bool {
        // L1 GATE: looking at screen + head forward
        let lookingAtScreen = gazeRatio >= gazeRatioThreshold && headFwdRatio >= headFwdRatioThreshold
        if !lookingAtScreen { return false }

        // Hard NO: silent mouth but audio active → someone else
        if jawStd < 0.004 && volAvg > 0.02 && pearson < 0.1 { return false }

        // L2 MAIN: jaw activity
        let strongJaw = maxJaw >= jawActivityThreshold
        let mediumJaw = jawStd >= jawStdThreshold && volStd > volActiveThreshold
        var verdict = strongJaw || mediumJaw

        // L3 PEARSON: filter false positives
        if verdict && pearson < -0.2 && volStd > 0.003 {
            verdict = false
        }
        // L3 PEARSON: rescue quiet speech
        if !verdict && pearson >= 0.35 && volStd > 0.003 && jawStd > 0.003 {
            verdict = true
        }
        return verdict
    }

    // MARK: - Sentence-level vote

    private func applySentenceVote(_ rows: [TokenRow]) -> [TokenRow] {
        guard rows.count >= 2 else {
            lastVerdict = .zero
            return rows
        }
        let userCount = rows.filter { $0.isUser }.count
        let ratio = Float(userCount) / Float(rows.count)
        let peak = rows.map { $0.fusedScore }.max() ?? 0

        var longest = 0, cur = 0
        for r in rows {
            if r.isUser { cur += 1; longest = max(longest, cur) } else { cur = 0 }
        }
        let spanRatio = Float(longest) / Float(rows.count)

        let hits = [ratio >= 0.40, peak >= 0.40, spanRatio >= 0.30].filter { $0 }.count
        let dominant = hits >= 2

        lastVerdict = Verdict(userRatio: ratio, peakScore: peak, longestSpanRatio: spanRatio, isUserDominant: dominant)

        guard dominant else { return rows }

        return rows.map { row in
            if row.isUser { return row }
            let lookingAtScreen = row.gazeRatio >= gazeRatioThreshold && row.headFwdRatio >= headFwdRatioThreshold
            if !lookingAtScreen { return row }
            // REQUIRED: mouth must have moved. Gaze+head-forward alone mean
            // "watching", not "speaking" — without this someone else's speech
            // leaks into the user sentence whenever the user looks at the screen.
            let jawMoved = row.maxJaw >= jawActivityThreshold || row.jawStd >= jawStdThreshold
            if !jawMoved { return row }
            // Keep as non-user if clearly not: silent mouth + loud audio.
            let clearlyNot = row.jawStd < 0.003 && row.avgVol > 0.03
            if clearlyNot { return row }
            return TokenRow(
                text: row.text, startTime: row.startTime, endTime: row.endTime,
                avgJaw: row.avgJaw, maxJaw: row.maxJaw, jawStd: row.jawStd,
                avgVol: row.avgVol, maxVol: row.maxVol, volStd: row.volStd,
                localPearson: row.localPearson, fusedScore: row.fusedScore,
                gazeRatio: row.gazeRatio, headFwdRatio: row.headFwdRatio,
                sampleCount: row.sampleCount, effectiveWindow: row.effectiveWindow,
                isUser: true, filledBySentence: true, isFinal: row.isFinal
            )
        }
    }

    // MARK: - Reconstruction

    private func presenceHits(_ row: TokenRow) -> Int {
        var hits = 0
        if row.maxJaw >= jawActivityThreshold || row.jawStd >= jawStdThreshold { hits += 1 }
        if row.gazeRatio >= gazeRatioThreshold { hits += 1 }
        if row.headFwdRatio >= headFwdRatioThreshold { hits += 1 }
        return hits
    }

    private func mouthMoved(_ row: TokenRow) -> Bool {
        row.maxJaw >= jawActivityThreshold || row.jawStd >= jawStdThreshold
    }

    private func reconstructSentence(volatile: [TokenRow], finalBatch: [TokenRow]) -> String {
        if !volatile.isEmpty {
            return volatile.filter { $0.isUser }.map(\.text).joined()
        }
        var tail: [TokenRow] = []
        var budget = gapBudgetPerSentence
        for row in finalBatch.reversed() {
            if row.isUser {
                tail.insert(row, at: 0)
                continue
            }
            if tail.isEmpty {
                continue
            }
            guard mouthMoved(row) else { break }
            let cost = 3 - presenceHits(row)
            if cost <= budget {
                tail.insert(row, at: 0)
                budget -= cost
            } else {
                break
            }
        }
        return tail.map(\.text).joined()
    }
}

#endif
