"""Select a CLEAN canonical-market layer from the permissive pool (artifacts/stageB/themes_market.parquet).

    python -m experiments.stageB_select_markets

The permissive (skip-PMI) pool has 69k terms -- too noisy. But it CONTAINS the real markets with their as-of
emergence signal. Here we pick the ~50 canonical AI/ML markets by NAME (a priori domain vocabulary, chosen
before seeing any money outcome -> not fishing), taking the highest-volume phrase variant per market as its
representative. Output: themes_market_clean.parquet (drop-in for stageB_bridge via THEMES_FILE).
"""
from __future__ import annotations
import pathlib
import re

import pandas as pd

SB = pathlib.Path("artifacts/stageB")

# canonical AI/ML markets -> regex over theme_id (a priori vocabulary; NOT chosen from outcomes)
MARKETS = {
    "large language models": r"large language model|\bllm\b|\bllms\b",
    "foundation models": r"foundation model",
    "diffusion models": r"diffusion model|denoising diffusion",
    "generative adversarial networks": r"generative adversarial",
    "graph neural networks": r"graph neural|\bgnn\b",
    "reinforcement learning": r"reinforcement learning",
    "rlhf": r"human feedback|\brlhf\b",
    "federated learning": r"federated learning",
    "self-supervised learning": r"self-supervised",
    "contrastive learning": r"contrastive learning",
    "vision transformers": r"vision transformer|\bvit\b",
    "transformers": r"^transformers?$|transformer architecture|transformer model",
    "object detection": r"object detection",
    "semantic segmentation": r"semantic segmentation|image segmentation",
    "image generation": r"image generation|text-to-image|text to image",
    "speech recognition": r"speech recognition|\basr\b",
    "speech synthesis": r"text-to-speech|speech synthesis|\btts\b",
    "machine translation": r"machine translation",
    "question answering": r"question answering",
    "named entity recognition": r"named entity",
    "recommender systems": r"recommender|recommendation system",
    "information retrieval": r"information retrieval|dense retrieval",
    "retrieval augmented generation": r"retrieval.augmented|\brag\b",
    "knowledge graphs": r"knowledge graph",
    "knowledge distillation": r"knowledge distillation",
    "neural architecture search": r"neural architecture|\bnas\b",
    "in-context learning": r"in-context",
    "prompt engineering": r"prompt tuning|prompt engineering|prompting",
    "chain-of-thought": r"chain-of-thought|chain of thought",
    "multimodal learning": r"multimodal|multi-modal",
    "vision-language models": r"vision-language|vision language",
    "autonomous driving": r"autonomous driving|self-driving|autonomous vehicle",
    "imitation/robot learning": r"imitation learning|robot learning|robotic manipulation",
    "out-of-distribution/anomaly": r"out-of-distribution|anomaly detection|\bood\b",
    "time series forecasting": r"time series",
    "adversarial robustness": r"adversarial (example|robust|attack|training|perturbation)",
    "differential privacy": r"differential privacy",
    "model compression": r"model compression|\bquantization\b|network pruning",
    "drug discovery": r"drug discovery|molecular property|molecule generation",
    "protein structure": r"protein structure|protein folding",
    "medical imaging": r"medical imaging|medical image",
    "explainable ai": r"explainab|interpretab|\bxai\b",
    "neural radiance fields": r"neural radiance|\bnerf\b",
    "point cloud / 3d": r"point cloud|3d reconstruction",
    "action/video recognition": r"action recognition|video understanding",
    "continual learning": r"continual learning|lifelong learning|catastrophic forgetting",
    "meta / few-shot learning": r"meta-learning|few-shot",
    "sentiment analysis": r"sentiment analysis",
    "generative models": r"generative model|variational autoencoder|\bvae\b",
    "llm agents": r"\bagent\b|agentic|autonomous agent",
}


def main():
    pool = pd.read_parquet(SB / "themes_market.parquet")
    picks = []
    for market, pat in MARKETS.items():
        rx = re.compile(pat, re.I)
        cand = pool[pool["theme_id"].str.contains(rx, regex=True)]
        if not len(cand):
            print(f"   [miss] {market}")
            continue
        rep = cand.sort_values("n_papers_asof", ascending=False).iloc[0].copy()  # highest-volume variant
        rep["market"] = market
        picks.append(rep)
    clean = pd.DataFrame(picks).drop_duplicates("theme_id").reset_index(drop=True)
    clean.to_parquet(SB / "themes_market_clean.parquet")
    print(f"\n[select] {len(clean)}/{len(MARKETS)} canonical markets found -> themes_market_clean.parquet")
    for _, r in clean.sort_values("sig_score", ascending=False).iterrows():
        print(f"   [{r['sig_score']:.2f}] {r['market']:28s} <- '{r['theme_id']}' ({r['label']}, {r['n_papers_asof']} papers as-of)")


if __name__ == "__main__":
    main()
