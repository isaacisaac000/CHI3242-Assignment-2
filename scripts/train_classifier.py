"""Train a Douban review sentiment classifier (jieba + TF-IDF + LogisticRegression)."""

import csv
import json
import logging
import os
from collections import Counter

import jieba
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "corpora/douban.json"
MODEL_PATH = "outputs/sentiment_model.joblib"
FEATURES_CSV = "outputs/top200_word_features.csv"
PLOT_PATH = "outputs/word_zscores.png"
RANDOM_STATE = 42
TOP_N_WEIGHTS = 20
TOP_N_COMMON = 200


def load_reviews(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_and_relabel(reviews):
    filtered = []
    for r in reviews:
        if r["label"] in (4, 5):
            filtered.append({"text": r["text"], "label": 1})
        elif r["label"] in (1, 2):
            filtered.append({"text": r["text"], "label": 0})
    return filtered


def downsample_to_min_class(texts, labels, rng):
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    n = min(len(v) for v in by_label.values())
    keep = []
    for y in sorted(by_label):
        idx = by_label[y]
        rng.shuffle(idx)
        keep.extend(idx[:n])
    keep.sort()
    return [texts[i] for i in keep], [labels[i] for i in keep]


def split_train_test(texts, labels):
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    train_idx, test_idx = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    for y in sorted(by_label):
        idx = np.array(by_label[y])
        rng.shuffle(idx)
        cut = int(round(0.8 * len(idx)))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_texts, train_labels, test_texts, test_labels


def segment(texts):
    return [[t.strip() for t in jieba.lcut(text) if t.strip()] for text in texts]


def top_weight_table(feature_names, coefs, vocab, X_test, test_labels, n):
    order = np.argsort(coefs)
    pos_idx = order[-n:][::-1]
    neg_idx = order[:n]
    pos_mask = np.array(test_labels) == 1
    neg_mask = ~pos_mask
    pos_means = np.asarray(X_test[pos_mask].mean(axis=0)).ravel()
    neg_means = np.asarray(X_test[neg_mask].mean(axis=0)).ravel()

    def rows(idx):
        out = []
        for i in idx:
            w = feature_names[i]
            out.append((w, coefs[i], pos_means[i], neg_means[i]))
        return out

    return rows(pos_idx), rows(neg_idx)


def print_weight_table(title, table):
    print(f"\n{title}")
    print(f"{'word':<12}{'weight':>10}{'tfidf pos':>12}{'tfidf neg':>12}")
    for w, c, tp, tn in table:
        print(f"{w:<12}{c:>10.4f}{tp:>12.6f}{tn:>12.6f}")


def top_common_word_features(train_tokens, train_labels):
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)
    top_words = [w for w, _ in counter.most_common(TOP_N_COMMON)]
    top_index = {w: j for j, w in enumerate(top_words)}
    freq = np.zeros((len(train_tokens), len(top_words)))
    for i, toks in enumerate(train_tokens):
        total = len(toks)
        if total == 0:
            continue
        for w, c in Counter(toks).items():
            j = top_index.get(w)
            if j is not None:
                freq[i, j] = c / total
    labels = np.array(train_labels)
    pos_mask = labels == 1
    mu = freq.mean(axis=0)
    sigma = freq.std(axis=0)
    sigma[sigma == 0] = 1e-12
    freq_pos = freq[pos_mask].mean(axis=0)
    freq_neg = freq[~pos_mask].mean(axis=0)
    z_pos = (freq_pos - mu) / sigma
    z_neg = (freq_neg - mu) / sigma
    return top_words, freq_pos, freq_neg, z_pos, z_neg


def save_features_csv(top_words, freq_pos, freq_neg, z_pos, z_neg):
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "freq_pos", "freq_neg", "z_pos", "z_neg"])
        for i, w in enumerate(top_words):
            writer.writerow(
                [w, f"{freq_pos[i]:.6f}", f"{freq_neg[i]:.6f}", f"{z_pos[i]:.4f}", f"{z_neg[i]:.4f}"]
            )


def plot_zscores(top_words, z_pos, z_neg, n=15):
    diff = z_pos - z_neg
    order = np.argsort(diff)
    sel = np.concatenate([order[-n:][::-1], order[:n][::-1]])
    labels = [top_words[i] for i in sel]
    zp = z_pos[sel]
    zn = z_neg[sel]
    y = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(y + 0.2, zp, height=0.4, color="#2e7d32", label="z-score positive class")
    ax.barh(y - 0.2, zn, height=0.4, color="#c62828", label="z-score negative class")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z-score of word relative frequency")
    ax.set_title(f"Class-distinctive usage of {TOP_N_COMMON}-most-common words")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def print_distinctive_words(top_words, z_pos, z_neg, n=15):
    diff = z_pos - z_neg
    order = np.argsort(diff)
    pos_sel = order[-n:][::-1]
    neg_sel = order[:n]
    print(f"\nMost positive-distinctive of the top-{TOP_N_COMMON} common words (by z-score gap):")
    print(f"{'word':<12}{'z pos':>9}{'z neg':>9}")
    for i in pos_sel:
        print(f"{top_words[i]:<12}{z_pos[i]:>9.2f}{z_neg[i]:>9.2f}")
    print(f"\nMost negative-distinctive of the top-{TOP_N_COMMON} common words (by z-score gap):")
    print(f"{'word':<12}{'z pos':>9}{'z neg':>9}")
    for i in neg_sel:
        print(f"{top_words[i]:<12}{z_pos[i]:>9.2f}{z_neg[i]:>9.2f}")


def main():
    os.makedirs("outputs", exist_ok=True)
    jieba.setLogLevel(logging.WARNING)

    reviews = load_reviews(DATA_PATH)
    print(f"Loaded {len(reviews)} reviews from {DATA_PATH}")

    filtered = filter_and_relabel(reviews)
    labels_all = [r["label"] for r in filtered]
    counts = Counter(labels_all)
    print(f"After filtering: {len(filtered)} reviews (positive={counts[1]}, negative={counts[0]})")

    rng = np.random.default_rng(RANDOM_STATE)
    texts, labels = downsample_to_min_class(
        [r["text"] for r in filtered], labels_all, rng
    )
    balanced_counts = Counter(labels)
    print(
        f"After balancing: {len(texts)} reviews "
        f"(positive={balanced_counts[1]}, negative={balanced_counts[0]})"
    )

    train_texts, train_labels, test_texts, test_labels = split_train_test(texts, labels)
    print(
        f"Train: {len(train_texts)} (pos={sum(train_labels)}, neg={len(train_labels) - sum(train_labels)}), "
        f"Test: {len(test_texts)} (pos={sum(test_labels)}, neg={len(test_labels) - sum(test_labels)})"
    )

    print("Segmenting with jieba...")
    train_tokens = segment(train_texts)
    test_tokens = segment(test_texts)
    train_docs = [" ".join(toks) for toks in train_tokens]
    test_docs = [" ".join(toks) for toks in test_tokens]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split, lowercase=False, token_pattern=None
    )
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    print(f"TF-IDF matrix: train {X_train.shape}, test {X_test.shape}")

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    model.fit(X_train, train_labels)

    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
    print(f"Saved vectorizer + model to {MODEL_PATH}")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    pos_table, neg_table = top_weight_table(
        feature_names, coefs, vectorizer.vocabulary_, X_test, test_labels, TOP_N_WEIGHTS
    )
    print_weight_table(f"Top {TOP_N_WEIGHTS} most positive-weight words:", pos_table)
    print_weight_table(f"Top {TOP_N_WEIGHTS} most negative-weight words:", neg_table)

    top_words, freq_pos, freq_neg, z_pos, z_neg = top_common_word_features(
        train_tokens, train_labels
    )
    save_features_csv(top_words, freq_pos, freq_neg, z_pos, z_neg)
    print(f"Saved top-{TOP_N_COMMON} word frequencies and z-scores to {FEATURES_CSV}")
    print_distinctive_words(top_words, z_pos, z_neg)
    plot_zscores(top_words, z_pos, z_neg)
    print(f"Saved z-score plot to {PLOT_PATH}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    print("\n=== Evaluation on test set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
    print(
        "\n"
        + classification_report(
            test_labels,
            y_pred,
            target_names=["negative (1-2 star)", "positive (4-5 star)"],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
