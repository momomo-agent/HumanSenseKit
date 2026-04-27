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

        // MARK: Confidence decomposition (4/27)
        // Independent of the boolean `isUser` verdict, these numeric signals
        // let the UI (and downstream logic) reason about *how* confident we
        // are that this token is the user speaking. Useful when the hard
        // gate says yes (mouth open, looking, audio present) but the audio
        // might actually be someone else.

        /// [0, 1] — attention gate (gaze + head). Multiplicative; if the user
        /// is clearly not looking at the device, overall confidence is zero
        /// regardless of mouth/audio coupling.
        public let gateScore: Float
        /// [0, 1] — mouth motion quality. Rewards open-and-varying jaw
        /// (speech-like), penalizes static/barely-moving mouth.
        public let mouthScore: Float
        /// [0, 1] — jaw-audio synchrony. The key "is it really you speaking"
        /// signal. Combines Pearson correlation, amplitude proportionality
        /// (jawStd ∼ volStd), and mouth motion presence. Low when audio is
        /// loud but jaw barely moves (someone else's voice).
        public let syncScore: Float
        /// [0, 1] — composed confidence. `gate * (0.35*mouth + 0.65*sync)`.
        /// High only when all three cooperate.
        public let userConfidence: Float

        /// Span-level verdict derived from running a Schmitt-trigger (double
        /// threshold + gap tolerance + short-span drop) over a temporally
        /// smoothed sequence of `userConfidence`. True when this token falls
        /// inside a contiguous user span, regardless of its own momentary
        /// conf value — so a single low-conf blip in the middle of a real
        /// utterance is NOT cut out.
        ///
        /// Computed and filled in after all tokens are recorded; the row
        /// itself defaults to false on construction.
        public var isUserWithConfidence: Bool = false
    }

    /// A contiguous run of tokens attributed to the user by the confidence
    /// span extractor. Use these instead of individual `isUser` flags when
    /// you need 'what did the user actually say in this chunk of audio'.
    public struct UserSpan: Sendable {
        public let startTime: Double       // wall-clock seconds
        public let endTime: Double         // wall-clock seconds
        public let tokenIndices: [Int]     // indices into the input row array
        public let text: String            // joined text of the tokens
        public let avgConfidence: Float    // mean userConfidence across tokens
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
    /// Symmetric margin (seconds) added to every token's audioTimeRange before
    /// sampling jaw/vol/gaze signals. STT tokenization timestamps can drift
    /// ~50-100ms from the actual jaw motion peak, and per-token ranges tend
    /// to be short (100-300ms), so without this margin `maxJaw` frequently
    /// misses the peak even when the user's mouth clearly opened during the
    /// word (observed: volatile whole-sentence row reports maxJaw=0.6, final
    /// per-character rows all report 0.11). The margin adds 2×marginSeconds
    /// to every matched window so adjacent tokens overlap slightly, which
    /// mirrors the continuity of real speech while preserving token-level
    /// attribution. 0 disables.
    public var tokenAudioRangeMarginSeconds: Double = 0.10
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

    // MARK: - Confidence span extraction (Schmitt trigger)
    //
    // These parameters control how `isUserWithConfidence` and `extractUserSpans`
    // convert the per-token `userConfidence` sequence into connected user
    // spans. Goals: don't cut the middle of a real utterance, don't include
    // foreign speech at either end.

    /// Half-width of the temporal smoothing window (seconds) applied to
    /// `userConfidence` before thresholding. Neighbors within this window
    /// contribute an exponentially decayed weight (see `confSmoothSigmaSec`).
    /// 0.20s matches the coarticulation timescale — a speaker's mouth-audio
    /// coupling doesn't fluctuate faster than that.
    public var confSmoothWindowSec: Double = 0.20
    /// Decay sigma for smoothing weights. Bigger = more smoothing.
    public var confSmoothSigmaSec: Double = 0.10
    /// Enter-user threshold. Smoothed conf must exceed this to start a span.
    public var confEnterThreshold: Float = 0.55
    /// Exit-user threshold (below this → start counting the gap). Must be < enter.
    public var confExitThreshold: Float = 0.30
    /// Max gap (seconds) allowed inside a user span before we close it.
    /// Lets breathing / pauses / single-token blips stay inside the span.
    public var confGapToleranceSec: Double = 0.15
    /// Spans shorter than this are discarded (noise / single-char false fires).
    public var confMinSpanDurationSec: Double = 0.30
    /// Two spans separated by less than this are merged into one.
    public var confSpanMergeGapSec: Double = 0.40

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

        // Run span extraction across finalized+volatile and write the
        // `isUserWithConfidence` flag back into both arrays. Spans need
        // to see across batch boundaries or single-batch edge tokens
        // would always fall outside their smoothed span.
        applyConfidenceSpans()

        tokens = finalizedTokens + volatileTokens

        userSentence = reconstructSentence(volatile: volatileTokens, finalBatch: lastFinalBatch)
    }

    /// Run the Schmitt-trigger span extractor over all recorded tokens
    /// and write `isUserWithConfidence` back into `finalizedTokens` and
    /// `volatileTokens`. Kept internal to `recordTokens`.
    private func applyConfidenceSpans() {
        let all = finalizedTokens + volatileTokens
        guard !all.isEmpty else { return }
        let flags = computeConfidenceSpanFlags(for: all)
        // Write back. finalizedTokens first N, then volatileTokens.
        let nFinal = finalizedTokens.count
        for i in 0..<nFinal {
            finalizedTokens[i].isUserWithConfidence = flags[i]
        }
        for j in 0..<volatileTokens.count {
            volatileTokens[j].isUserWithConfidence = flags[nFinal + j]
        }
    }

    /// Public API: extract the user's spoken spans from an arbitrary row
    /// sequence (e.g. a transcript chunk). Stateless, so callers can feed
    /// it any subset of rows (including results from another reconstructor).
    public func extractUserSpans(from rows: [TokenRow]) -> [UserSpan] {
        let flags = computeConfidenceSpanFlags(for: rows)
        var spans: [UserSpan] = []
        var i = 0
        while i < rows.count {
            guard flags[i] else { i += 1; continue }
            let start = i
            while i < rows.count && flags[i] { i += 1 }
            let end = i - 1
            let tokens = Array(rows[start...end])
            let text = tokens.map { $0.text }.joined()
            let avgConf = tokens.isEmpty ? 0 : tokens.map { $0.userConfidence }.reduce(0, +) / Float(tokens.count)
            spans.append(UserSpan(
                startTime: tokens.first!.startTime,
                endTime: tokens.last!.endTime,
                tokenIndices: Array(start...end),
                text: text,
                avgConfidence: avgConf
            ))
        }
        return spans
    }

    // MARK: - Schmitt-trigger core

    /// Core of the confidence-span algorithm. Returns a flag per input
    /// row indicating whether that row lives inside a recognised user
    /// span. Steps:
    ///   1. Temporal smoothing of userConfidence with exponential weights
    ///   2. Two-threshold hysteresis (enter >= 0.55, exit < 0.30)
    ///   3. Gap tolerance inside a span (<= 150ms default — absorbs
    ///      breathing, micro-pauses, one-token conf dips)
    ///   4. Drop spans shorter than 300ms (noise / single-char blips)
    ///   5. Merge spans separated by less than 400ms
    private func computeConfidenceSpanFlags(for rows: [TokenRow]) -> [Bool] {
        let n = rows.count
        guard n > 0 else { return [] }

        // --- Step 1: temporal smoothing of userConfidence ---
        // Use midpoint time per token so token length doesn't distort weights.
        var midpoints: [Double] = []
        midpoints.reserveCapacity(n)
        for r in rows { midpoints.append((r.startTime + r.endTime) * 0.5) }

        var smoothed = [Float](repeating: 0, count: n)
        let window = confSmoothWindowSec
        let sigma = max(confSmoothSigmaSec, 0.01)
        for i in 0..<n {
            var wsum: Double = 0
            var wconf: Double = 0
            // Walk outward from i while within window. O(n) on small n; for
            // larger n we could bisect, but typical input is 5-50 tokens.
            for k in 0..<n {
                let dt = abs(midpoints[k] - midpoints[i])
                if dt > window { continue }
                let w = exp(-dt / sigma)
                wsum += w
                wconf += Double(rows[k].userConfidence) * w
            }
            smoothed[i] = wsum > 0 ? Float(wconf / wsum) : rows[i].userConfidence
        }

        // --- Step 2+3: Schmitt trigger with gap tolerance ---
        // We produce raw spans [startIdx, endIdx] (inclusive).
        struct RawSpan { var start: Int; var end: Int }
        var rawSpans: [RawSpan] = []
        var inSpan = false
        var spanStart = 0
        var lastInIdx = 0             // last index that was clearly inside
        var belowStartTime: Double? = nil

        for i in 0..<n {
            let s = smoothed[i]
            let t = midpoints[i]

            if !inSpan {
                if s >= confEnterThreshold {
                    inSpan = true
                    spanStart = i
                    lastInIdx = i
                    belowStartTime = nil
                }
            } else {
                if s < confExitThreshold {
                    // Start of a possible gap.
                    if belowStartTime == nil { belowStartTime = t }
                    if let bs = belowStartTime, t - bs > confGapToleranceSec {
                        // Gap too long → close span at lastInIdx.
                        rawSpans.append(RawSpan(start: spanStart, end: lastInIdx))
                        inSpan = false
                        belowStartTime = nil
                    }
                } else {
                    // Back above exit threshold — reset gap, extend span.
                    belowStartTime = nil
                    lastInIdx = i
                }
            }
        }
        if inSpan {
            rawSpans.append(RawSpan(start: spanStart, end: lastInIdx))
        }

        // --- Step 4: drop short spans ---
        var kept: [RawSpan] = []
        for sp in rawSpans {
            let dur = rows[sp.end].endTime - rows[sp.start].startTime
            if dur >= confMinSpanDurationSec { kept.append(sp) }
        }

        // --- Step 5: merge close spans ---
        var merged: [RawSpan] = []
        for sp in kept {
            if var last = merged.last,
               rows[sp.start].startTime - rows[last.end].endTime < confSpanMergeGapSec {
                last.end = sp.end
                merged[merged.count - 1] = last
            } else {
                merged.append(sp)
            }
        }

        // --- Build flags ---
        var flags = [Bool](repeating: false, count: n)
        for sp in merged {
            for k in sp.start...sp.end { flags[k] = true }
        }
        return flags
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

    /// Attribute an arbitrary time range against the tokens this
    /// reconstructor has seen.
    ///
    /// IMPORTANT: `start` and `end` are WALL-CLOCK seconds (Unix epoch),
    /// to match what `TokenRow.startTime / endTime` store internally
    /// (which are `audioStreamStartTime + token.startTime`).
    ///
    /// Earlier versions of this API claimed the parameters were
    /// "audio-stream-relative", but the comparison inside was always
    /// done against wall-clock row times — so passing audio-offset
    /// seconds would guarantee zero overlap and the caller would always
    /// see `hasCoverage=false`. This was the root cause of visual-talk-ios
    /// showing `isFromUser=false` in segments where demo's Tokens tab
    /// clearly showed all-green user tokens with ratio=100%.
    ///
    /// Used by `STTManager.rebuildSegments()` to transparently upgrade
    /// `SpeechSegment.isFromUser` without changing the builder itself.
    ///
    /// - Parameters:
    ///   - start: wall-clock seconds (Unix epoch, nil → hasCoverage=false)
    ///   - end:   wall-clock seconds (Unix epoch, nil → hasCoverage=false)
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

    /// Returns the concatenated text of tokens whose time range overlaps
    /// [start, end] AND whose per-token verdict is `isUser == true`.
    ///
    /// IMPORTANT: `start` and `end` are WALL-CLOCK seconds (Unix epoch),
    /// matching `TokenRow.startTime / endTime` semantics. See
    /// `attribution(for:end:)` for the full rationale.
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
        // Apply a symmetric margin to the token's audio range before sampling.
        // STT token timestamps are best-effort: the actual jaw motion peak
        // for a character can sit 50-100ms outside the reported range, and
        // per-character final ranges are short enough (100-300ms) that
        // `jaws.max()` routinely misses the peak without this margin.
        let margin = tokenAudioRangeMarginSeconds
        var expand: Double = 0
        let baseStart = wallStart - margin
        let baseEnd = wallEnd + margin
        var matched = samples.filter { $0.ts >= baseStart && $0.ts <= baseEnd }
        while matched.count < minSampleCount && expand < maxWindowMs {
            expand += windowExpandMs
            let s = baseStart - expand
            let e = baseEnd + expand
            matched = samples.filter { $0.ts >= s && $0.ts <= e }
        }

        // Gross-mismatch fallback: if expansion failed to find ANY samples,
        // the token's wall-clock timeline is completely out of sync with the
        // sample ring buffer (typically caused by app backgrounding + audio
        // engine restart without re-stamping audioStreamStartTime, so token
        // times point at a stale base). In that case fall back to the most
        // recent samples within a reasonable window so we still have signals
        // to classify the token, rather than producing an all-zero row that
        // would be wrongly judged non-user.
        if matched.isEmpty, let latest = samples.last {
            let fallbackWindow: Double = 1.0 // last 1s of samples
            matched = samples.filter { $0.ts >= latest.ts - fallbackWindow }
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
        let maxJawVal = jaws.max() ?? 0
        let gateScore = computeGateScore(gazeRatio: gazeRatio, headFwdRatio: headFwdRatio)
        let mouthScore = computeMouthScore(maxJaw: maxJawVal, jawStd: jawStdVal)
        let syncScore = computeSyncScore(
            pearson: pearson,
            jawStd: jawStdVal,
            volStd: volStdVal,
            volAvg: volAvg,
            maxJaw: maxJawVal
        )
        let userConfidence = composeConfidence(gate: gateScore, mouth: mouthScore, sync: syncScore)

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
            isFinal: isFinal,
            gateScore: gateScore,
            mouthScore: mouthScore,
            syncScore: syncScore,
            userConfidence: userConfidence
        )
    }

    // MARK: - Confidence decomposition

    /// L1 attention gate. If the user is not looking at the screen at all
    /// (gaze + head both close to zero), this token cannot plausibly be
    /// directed at the device, regardless of mouth/audio coupling.
    ///
    /// Soft ramp instead of a binary gate: a token where the user glanced
    /// away mid-word should not hard-zero, so we smoothstep around the
    /// existing ratio thresholds and take the product of gaze and head
    /// factors (both must be reasonably present for the gate to open).
    private func computeGateScore(gazeRatio: Float, headFwdRatio: Float) -> Float {
        let gaze = smoothstep(edge0: 0.05, edge1: gazeRatioThreshold + 0.2, x: gazeRatio)
        let head = smoothstep(edge0: 0.05, edge1: headFwdRatioThreshold + 0.2, x: headFwdRatio)
        return gaze * head
    }

    /// L2 mouth motion. High when the mouth opens distinctly AND varies in
    /// amplitude (as in real speech), low when the mouth is nearly static
    /// (silence or external-speaker case) or barely open (chewing / breathing).
    private func computeMouthScore(maxJaw: Float, jawStd: Float) -> Float {
        let openness = smoothstep(edge0: 0.04, edge1: jawActivityThreshold + 0.08, x: maxJaw)
        let variation = smoothstep(edge0: jawStdThreshold * 0.5, edge1: jawStdThreshold * 2.5, x: jawStd)
        // Need BOTH amplitude and variation — a wide jaw that doesn't move
        // (yawn held open) scores low on variation and gets penalized.
        return sqrt(max(0, openness * variation))
    }

    /// L3 jaw-audio synchrony. The signature signal of "this token is the
    /// user speaking": when the user talks, the jaw opening and audio
    /// amplitude rise together (positive Pearson) AND mouth motion is in
    /// proportion to the audio loudness (jaw amplitude ∼ volume).
    ///
    /// The hardest false positive to kill is: someone else speaks, the user's
    /// mouth happens to be slightly open or moving for unrelated reasons.
    /// In that case Pearson will be near 0 or negative, and `jawStd/volStd`
    /// will be badly out of proportion. We detect both.
    private func computeSyncScore(
        pearson: Float,
        jawStd: Float,
        volStd: Float,
        volAvg: Float,
        maxJaw: Float
    ) -> Float {
        // No meaningful audio → no sync to measure; assume silent == could
        // still be user (e.g. the mouth motion IS the word). Neutral 0.5.
        if volStd < 0.002 && volAvg < 0.01 { return 0.5 }

        // Pearson: map [−1, +1] → [0, 1], with a bonus for positive
        // correlation and a penalty for negative (someone else's voice while
        // your jaw drifts the other way).
        let pearsonScore: Float = {
            if pearson >= 0.3 { return 1.0 }
            if pearson >= 0.0 { return 0.3 + pearson * 2.33 }   // 0.3 → 0.3+0.7
            return max(0, 0.3 + pearson * 0.5)                  // −0.6 → 0
        }()

        // Amplitude ratio: if audio is loud (high volStd) but mouth barely
        // moves (low jawStd), that is the classic "someone else speaking"
        // signature. Penalize.
        let ampRatio: Float = {
            if volStd < 0.003 { return 0.7 }   // quiet audio, neutral
            let expected = volStd * 15   // rough empirical proportion
            let deficit = max(0, expected - jawStd) / max(expected, 0.001)
            return max(0, 1 - deficit)
        }()

        // Require some jaw motion at all — a perfectly still mouth can't be
        // producing speech even if Pearson looks ok by chance.
        let motionPresence = smoothstep(edge0: 0.02, edge1: 0.10, x: maxJaw)

        return pearsonScore * 0.5 + ampRatio * 0.35 + motionPresence * 0.15
    }

    /// Compose the three sub-scores into a single [0, 1] confidence.
    /// gateScore is a multiplicative gate (can hard-zero), mouth+sync are
    /// additive with sync weighted heavier (it is the anti-impostor signal).
    private func composeConfidence(gate: Float, mouth: Float, sync: Float) -> Float {
        let inner = mouth * 0.35 + sync * 0.65
        return max(0, min(1, gate * inner))
    }

    private func smoothstep(edge0: Float, edge1: Float, x: Float) -> Float {
        guard edge1 > edge0 else { return x >= edge1 ? 1 : 0 }
        let t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)
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
                isUser: true, filledBySentence: true, isFinal: row.isFinal,
                gateScore: row.gateScore, mouthScore: row.mouthScore,
                syncScore: row.syncScore, userConfidence: row.userConfidence
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
