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

## Key Observations
1. gazeOnScreen is unreliable — reports 1.0 when not looking (file 01), 0.4-0.6 when looking (file 02)
2. jawDelta/jawVelocity don't distinguish user from background speakers
3. score (speaker embedding) has no clear separation
4. Final phase tends to be more conservative (fewer false positives) than streaming
