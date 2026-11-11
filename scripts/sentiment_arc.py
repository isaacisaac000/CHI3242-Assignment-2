"""Plot the sentiment arc of a novel using the trained Douban sentiment model."""

import csv
import os
import re

import jieba
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TEXT_PATH = "corpora/52_我城.txt"
MODEL_PATH = "outputs/sentiment_model.joblib"
CSV_PATH = "outputs/sentiment_arc.csv"
PLOT_PATH = "outputs/sentiment_arc.png"
TARGET_CHARS = 500
MA_WINDOW = 10
SENTENCE_RE = re.compile(r"[^。！？]*[。！？]")


def split_passages(text):
    matches = list(SENTENCE_RE.finditer(text))
    sentences = [m.group() for m in matches]
    last_end = matches[-1].end() if matches else 0
    remainder = text[last_end:]
    if remainder.strip():
        sentences.append(remainder)
    passages = []
    buf = []
    length = 0
    for s in sentences:
        buf.append(s)
        length += len(s)
        if length >= TARGET_CHARS:
            passages.append("".join(buf))
            buf = []
            length = 0
    if buf:
        tail = "".join(buf).strip()
        if tail:
            if len(tail) < TARGET_CHARS / 3 and passages:
                passages[-1] += tail
            else:
                passages.append(tail)
    return [p.strip() for p in passages if p.strip()]


def score_passages(passages, vectorizer, model):
    pos_i = list(model.classes_).index(1)
    scores = []
    for text in passages:
        tokens = " ".join(t.strip() for t in jieba.lcut(text) if t.strip())
        X = vectorizer.transform([tokens])
        p_pos = model.predict_proba(X)[0][pos_i]
        scores.append(float(p_pos - (1.0 - p_pos)))
    return scores


def preview(text, n=100):
    return re.sub(r"\s+", " ", text)[:n]


def main():
    os.makedirs("outputs", exist_ok=True)
    jieba.setLogLevel(60)

    with open(TEXT_PATH, encoding="utf-8-sig") as f:
        text = f.read()
    passages = split_passages(text)
    print(f"{TEXT_PATH}: {len(text)} characters -> {len(passages)} passages (~{len(text) / len(passages):.0f} chars each)")

    bundle = joblib.load(MODEL_PATH)
    vectorizer, model = bundle["vectorizer"], bundle["model"]
    scores = score_passages(passages, vectorizer, model)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["passage_number", "score", "text"])
        for i, (s, t) in enumerate(zip(scores, passages), 1):
            writer.writerow([i, f"{s:.4f}", t])
    print(f"Saved {len(passages)} scored passages to {CSV_PATH}")

    x = np.arange(1, len(scores) + 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, scores, color="#9e9e9e", linewidth=0.8, alpha=0.7, label="passage score")
    if len(scores) >= MA_WINDOW:
        ma = np.convolve(scores, np.ones(MA_WINDOW) / MA_WINDOW, mode="valid")
        ax.plot(np.arange(MA_WINDOW, len(scores) + 1), ma, color="#1565c0", linewidth=2,
                label=f"{MA_WINDOW}-passage moving average")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("passage number")
    ax.set_ylabel("score  (P(positive) − P(negative))")
    ax.set_title("《我城》(西西) — sentiment arc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")

    order = np.argsort(scores)
    print(f"\nMean score: {np.mean(scores):.4f}  min: {scores[order[0]]:.4f}  max: {scores[order[-1]]:.4f}")

    print("\n3 highest-scoring passages:")
    for i in order[::-1][:3]:
        print(f"  #{i + 1}  score={scores[i]:+.4f}  {preview(passages[i])}")
    print("\n3 lowest-scoring passages:")
    for i in order[:3]:
        print(f"  #{i + 1}  score={scores[i]:+.4f}  {preview(passages[i])}")


if __name__ == "__main__":
    main()
