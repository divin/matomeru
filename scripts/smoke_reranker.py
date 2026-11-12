#!/usr/bin/env python3
"""Smoke test: score the real profile against crafted on-topic and off-topic posts.

Checks that the reranker endpoint (RERANKER_URL, default
http://192.168.178.64:9093) ranks the on-topic posts above the off-topic
ones, i.e. would produce a sensible digest selection.

Usage: uv run python scripts/smoke_reranker.py
"""

import json
import sys
import urllib.request

from matomeru.config import load_app_config, render_profile_markdown

RERANKER_URL = "http://192.168.178.64:9093/v1/rerank"

CONFIG_PATH = "matomeru.toml"

# (expected, title) -- expected: "on" = on-topic, "off" = off-topic
POSTS = [
    ("on", "Released: Qwen3-14B GGUF quants tested on Intel Arc B60 - 20 t/s with 24 GB VRAM"),
    ("on", "ZFS 2.3 released: new RAIDz expansion finally stable, upgrade guide for existing pools"),
    ("on", "Piper TTS v2: real-time German voice synthesis on consumer GPUs, 200ms latency"),
    ("on", "Steam Machine teardown: what's inside, and how it compares to a DIY Arc B60 build"),
    ("on", "MicroDuck firmware update adds Python API for custom gripper behaviors"),
    ("on", "New EU-hosted cloud alternative to Google Drive, self-hostable with Docker"),
    ("on", "ETF Ausführungskosten ab 2026: neue Regulierung für EU-Investoren erklärt"),
    ("on", "Recommended graded readers for JLPT N4 learners - 2026 edition"),
    ("on", "Usenet provider blacklist update: which indexers still work after this week's DMCA wave"),
    ("on", "TabPFN v3: tabular foundation model beats XGBoost on small classification datasets"),
    ("off", "Best pasta carbonara recipe - the Roman way, no cream"),
    ("off", "My cat learned to open the fridge, AMA"),
    ("off", "NBA playoffs: dramatic game 7 overtime winner"),
    ("off", "Beginner guitar lesson: first three chords"),
    ("off", "Photos from my hike in Patagonia"),
    ("off", "How to fix a squeaky door hinge in 5 minutes"),
    ("off", "Celebrity gossip: couple announces engagement"),
    ("off", "Top 10 productivity apps for iOS"),
    ("off", "Sourdough starter troubleshooting: why is my bread flat"),
    ("off", "Politically neutral discussion of a new zoning law in a small US town"),
]

profile = render_profile_markdown(load_app_config(CONFIG_PATH).profile_sections)

payload = json.dumps(
    {"model": "reranker", "query": profile, "documents": [t for _, t in POSTS]}
).encode()

req = urllib.request.Request(
    RERANKER_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.load(resp)

results = {r["index"]: r["relevance_score"] for r in body["results"]}
assert len(results) == len(POSTS), "response integrity: missing scores"

on_scores = [results[i] for i, (exp, _) in enumerate(POSTS) if exp == "on"]
off_scores = [results[i] for i, (exp, _) in enumerate(POSTS) if exp == "off"]

print(f"{'rank':>4}  {'score':>9}  {'exp':>3}  title")
print("-" * 72)
for rank, (idx, score) in enumerate(sorted(results.items(), key=lambda kv: -kv[1]), 1):
    exp, title = POSTS[idx]
    print(f"{rank:>4}  {score:>9.3f}  {exp:>3}  {title[:58]}")

print("-" * 72)
print(f"on-topic:  min={min(on_scores):.3f}  max={max(on_scores):.3f}  mean={sum(on_scores)/len(on_scores):.3f}")
print(f"off-topic: min={min(off_scores):.3f}  max={max(off_scores):.3f}  mean={sum(off_scores)/len(off_scores):.3f}")

# Perfect separation: every on-topic post scores above every off-topic post.
worst_on = min(on_scores)
best_off = max(off_scores)
if worst_on > best_off:
    gap = worst_on - best_off
    print(f"\nPASS: perfect separation, gap={gap:.3f}")
else:
    print(f"\nFAIL: overlap (worst on-topic {worst_on:.3f} <= best off-topic {best_off:.3f})")
    sys.exit(1)
