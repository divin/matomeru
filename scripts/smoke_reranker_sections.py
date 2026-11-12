#!/usr/bin/env python3
"""Smoke test 2: per-section profile queries, max per post.

Compares the whole-profile query (flat separation observed) against
scoring each post against each profile section and taking the max.

Usage: uv run python scripts/smoke_reranker_sections.py
"""

import json
import sys
import urllib.request

RERANKER_URL = "http://192.168.178.64:9093/v1/rerank"

SECTIONS = [
    ("About me", "Data scientist working mostly with tabular data: classification models and A/B testing. Lives in Germany, learning Japanese (roughly JLPT N4 / CEFR B). Self-hosting enthusiast - services and AI run on my own hardware wherever possible."),
    ("Local AI", "I self-host local AI on an Intel Arc B60 (24 GB VRAM) with 64 GB system RAM. Most relevant: new general-purpose and coding models that realistically run on this hardware, quantization and optimization posts, and local-LLM ecosystem news that could affect my setup. I also follow coding-agent and LLM-agent developments. Additionally, I am interested in TTS and STT engines and models, especially ones good enough for real-time usage."),
    ("Homelab & self-hosting", "Ubuntu Server home server with a ZFS pool on several HDDs. Most relevant: news affecting ZFS/HDD storage and data durability, and new self-hosted services worth adopting. Hardware interest: Intel Arc GPUs and the Steam Machine - new releases, deals, optimizations."),
    ("Usenet", "I am a Usenet user. Most relevant: provider news, sales events and deals, indexers, and tooling or ecosystem changes that could affect my Usenet access or workflow."),
    ("Germany, finance & Japan", "German current events, political trends, and anything affecting life in Germany; also Japan-related news. Personal finance: tips to improve my savings rate, ETF portfolio topics, new regulations affecting EU investors. EU alternatives to US services, especially ones that fit my other interests (self-hosting, local AI). Learning Japanese: new apps, resources, tips and tricks. My level is roughly JLPT N4 / CEFR B, so beginner-to-intermediate content is most useful, and I am always on the lookout for book recommendations (graded readers, textbooks, native books suitable for that level)."),
    ("News & gaming", "World events that affect Germany or could affect me personally. Steam: hardware (Steam Machine) news, sales events, and releases relevant to my hardware interests."),
    ("Robotics", "I have (or am getting) a MicroDuck from Pollen Robotics. Posts or papers that improve the MicroDuck experience - setup, software, capabilities - are directly relevant."),
]

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

def rerank(query: str, documents: list[str]) -> list[float]:
    payload = json.dumps({"model": "reranker", "query": query, "documents": documents}).encode()
    req = urllib.request.Request(
        RERANKER_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    results = {r["index"]: r["relevance_score"] for r in body["results"]}
    assert len(results) == len(documents), "response integrity: missing scores"
    return [results[i] for i in range(len(documents))]

def report(name: str, scores: list[float]) -> None:
    worst_on = min(s for s, (exp, _) in zip(scores, POSTS) if exp == "on")
    best_off = max(s for s, (exp, _) in zip(scores, POSTS) if exp == "off")
    order = sorted(range(len(POSTS)), key=lambda i: -scores[i])
    print(f"\n=== {name} ===")
    print(f"{'rank':>4}  {'score':>8}  {'exp':>3}  title")
    for rank, i in enumerate(order, 1):
        exp, title = POSTS[i]
        print(f"{rank:>4}  {scores[i]:>8.3f}  {exp:>3}  {title[:58]}")
    on_mean = sum(s for s, (exp, _) in zip(scores, POSTS) if exp == "on") / 10
    off_mean = sum(s for s, (exp, _) in zip(scores, POSTS) if exp == "off") / 10
    print(f"on-topic mean={on_mean:.3f}  off-topic mean={off_mean:.3f}")
    if worst_on > best_off:
        print(f"PASS: perfect separation, gap={worst_on - best_off:.3f}")
    else:
        print(f"FAIL: overlap (worst on-topic {worst_on:.3f} <= best off-topic {best_off:.3f})")

# Variant A: whole profile as one query
from matomeru.config import load_app_config, render_profile_markdown
profile = render_profile_markdown(load_app_config("matomeru.toml").profile_sections)
report("A: whole profile", rerank(profile, [t for _, t in POSTS]))

# Variant B: per-section queries, max per post
scores = [float("-inf")] * len(POSTS)
for name, text in SECTIONS:
    for i, s in enumerate(rerank(text, [t for _, t in POSTS])):
        scores[i] = max(scores[i], s)
report("B: per-section max", scores)
sys.exit(0)
