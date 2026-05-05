# Test Data v3 — Per-Sentence Speaker Attribution

## Format
JSONL files exported from GazeSpeakerEngine debug log.
Each line: `{ sentenceId, phase, text, isUserSpeaker, gazeOnScreen, jawDelta, jawVelocity, score, headYaw, headPitch, faceDistance, timestamp }`

## Files

### 01-all-nonuser.jsonl
- 205 entries, 6 sentences
- **Ground truth: ALL non-user** (kenefe was not speaking to screen)
- Current result: Final 100% correct, Streaming has 25-39% false positives

### 02-mixed.jsonl
- 414 entries, 13 sentences
- **Ground truth:**
  - User sentences (kenefe speaking to screen):
    - Sentence 5: "你吃饭了吗" (mixed with background — only tail is user)
    - Sentence 7: "今天天气怎么样"
    - Sentence 9: "真的吗"
    - Sentence 10: "你看过吗"
    - Sentence 11: "这是什么"
    - Sentence 12: "风生水起"
  - Non-user sentences: 0, 1, 2, 3, 4, 6, 8
- Current result: Almost all marked non-user (false negatives for user sentences)
- Key issue: gazeOnScreen only 0.4-0.6 when user IS looking at screen

### 05-mixed-v4.jsonl
- From 2026-05-05 autoresearch v4 session
- See evaluate.py for ground truth

### 06-mixed-v5.jsonl
- From 2026-05-05 17:51 session (post circular-dependency fix, HumanSenseKit 4.25.0)
- **Ground truth:**
  - User sentence (kenefe speaking to screen):
    - Sentence 0: "你好啊" (streaming shows "说你好啊" — first char "说" is ambient)
  - Non-user sentences: 1, 2, 3, 4 (all ambient/TV/other people)
- Key issue: jawDelta high (0.48-0.57) on non-user sentences because kenefe's jaw moves slightly while listening. Streaming sentence-level classifier marks all as user.
- Per-char final correctly identifies boundaries (S0 final only has "你好啊")

## Key Observations
1. gazeOnScreen is unreliable — reports 1.0 when not looking (file 01), 0.4-0.6 when looking (file 02)
2. jawDelta/jawVelocity don't distinguish user from background speakers
3. score (speaker embedding) has no clear separation
4. Final phase tends to be more conservative (fewer false positives) than streaming
5. Streaming sentence-level classification is too coarse — per-char split needed to avoid showing ambient prefix/suffix
