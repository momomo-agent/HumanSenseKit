#!/usr/bin/env python3
"""Evaluate speaker attribution accuracy against v3 ground truth."""
import json, sys
from collections import defaultdict

# Ground truth: { filename: { sentenceId: is_user } }
GROUND_TRUTH = {
    "01-all-nonuser.jsonl": {},  # all non-user, empty = default non-user
    "02-mixed.jsonl": {5: True, 7: True, 9: True, 10: True, 11: True, 12: True},
}

def evaluate(filepath, truth):
    lines = open(filepath).readlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    
    sentences = defaultdict(list)
    for e in entries:
        sentences[e.get('sentenceId', 0)].append(e)
    
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    details = []
    
    for sid in sorted(sentences.keys()):
        tokens = sentences[sid]
        expected_user = truth.get(sid, False)
        
        for phase in ["streaming", "final"]:
            phase_tokens = [t for t in tokens if t.get("phase") == phase]
            if not phase_tokens:
                continue
            
            user_count = sum(1 for t in phase_tokens if t["isUserSpeaker"])
            total = len(phase_tokens)
            user_ratio = user_count / total if total else 0
            
            # Sentence-level: majority vote
            predicted_user = user_ratio > 0.5
            
            if expected_user and predicted_user:
                results["tp"] += 1
                verdict = "TP ✅"
            elif not expected_user and not predicted_user:
                results["tn"] += 1
                verdict = "TN ✅"
            elif not expected_user and predicted_user:
                results["fp"] += 1
                verdict = "FP ❌"
            else:
                results["fn"] += 1
                verdict = "FN ❌"
            
            text = ''.join(t['text'] for t in [t for t in tokens if t.get('phase') == 'final'][:20])
            avg_gaze = sum(t['gazeOnScreen'] for t in phase_tokens) / len(phase_tokens)
            details.append({
                "sid": sid, "phase": phase, "text": text[:30],
                "expected": "user" if expected_user else "non-user",
                "user_ratio": user_ratio, "verdict": verdict,
                "avg_gaze": avg_gaze
            })
    
    return results, details

def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    
    total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    
    for fname, truth in GROUND_TRUTH.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"⚠️  {fname} not found, skipping")
            continue
        
        results, details = evaluate(path, truth)
        print(f"\n{'='*60}")
        print(f"📄 {fname}")
        print(f"{'='*60}")
        
        for d in details:
            print(f"  S{d['sid']:2d} {d['phase']:9s} | {d['verdict']} | expect={d['expected']:8s} | user_ratio={d['user_ratio']:.0%} | gaze={d['avg_gaze']:.2f} | \"{d['text']}\"")
        
        for k in total:
            total[k] += results[k]
    
    print(f"\n{'='*60}")
    print(f"📊 TOTAL")
    print(f"{'='*60}")
    tp, tn, fp, fn = total["tp"], total["tn"], total["fp"], total["fn"]
    total_cases = tp + tn + fp + fn
    accuracy = (tp + tn) / total_cases if total_cases else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    print(f"  Accuracy:  {accuracy:.1%} ({tp+tn}/{total_cases})")
    print(f"  Precision: {precision:.1%} (TP={tp}, FP={fp})")
    print(f"  Recall:    {recall:.1%} (TP={tp}, FN={fn})")
    print(f"  F1:        {f1:.1%}")

if __name__ == "__main__":
    main()
