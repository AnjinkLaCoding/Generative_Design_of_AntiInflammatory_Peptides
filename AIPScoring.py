"""
Anti-Inflammatory Peptide (AIP) Scoring Module
================================================

Trains an anti-inflammatory activity classifier on a labeled AIP dataset
and uses it to rank candidates produced by the
APD3-anti-inflammatory-trained generator (amp_generator.py).

Unlike DBAASP (quantitative MIC against microbes), these datasets are
binary-labeled: a peptide is "AIP" (positive) if it's been shown to induce
anti-inflammatory cytokines (IL-10, IL-4, IL-13, IL-22, TGF-b, IFN-a/b),
and "non-AIP" (negative) otherwise. So this module works with simple
sequence + label pairs rather than concentration values.

Supported input formats (use whichever matches what you downloaded):
    1. Two separate FASTA files: one of positive (AIP) sequences, one of
       negative (non-AIP) sequences -- the typical PreAIP/AIPpred export.
    2. A single CSV with a sequence column and a label column.
"""

import numpy as np
from Utils import (
    AA_SET,
    net_charge,
    mean_hydrophobicity,
    hydrophobic_moment,
    load_fasta,
)

# ---------------------------------------------------------------------------
# 1. Loading labeled AIP data
# ---------------------------------------------------------------------------
 
def load_labeled_fasta_pair(positive_fasta, negative_fasta, min_len=2, max_len=50):
    """Load a positive (AIP) and negative (non-AIP) FASTA file pair into a
    single labeled dataset. This matches the typical PreAIP/AIPpred export
    format (separate files per class).
    """
    pos_seqs = load_fasta(positive_fasta, min_len=min_len, max_len=max_len)
    neg_seqs = load_fasta(negative_fasta, min_len=min_len, max_len=max_len)
    data = [(s, 1) for s in pos_seqs] + [(s, 0) for s in neg_seqs]
    print(f"Loaded {len(pos_seqs)} positive (AIP) and {len(neg_seqs)} negative (non-AIP) sequences")
    return data
 
 
CANDIDATE_SEQ_COLUMNS = ["sequence", "Sequence", "SEQUENCE", "seq", "peptide"]
CANDIDATE_LABEL_COLUMNS = ["label", "Label", "class", "Class", "activity", "is_aip"]
 
 
# ---------------------------------------------------------------------------
# 2. Feature extraction + classifier
#    (same physicochemical feature set as dbaasp_scoring.py, so results are
#    comparable if you ever run both scorers side by side)
# ---------------------------------------------------------------------------
 
def featurize(seq):
    length = len(seq)
    comp = {aa: seq.count(aa) / length for aa in sorted(AA_SET)}
    return [
        length,
        net_charge(seq),
        mean_hydrophobicity(seq),
        hydrophobic_moment(seq),
        *comp.values(),
    ]
 
 
def train_aip_classifier(labeled_data, test_size=0.2, seed=0):
    """Train a RandomForest classifier to distinguish AIP from non-AIP
    sequences. Returns (model, holdout_accuracy, holdout_auc).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
 
    X = np.array([featurize(s) for s, _ in labeled_data])
    y = np.array([label for _, label in labeled_data])
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
 
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
 
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
 
    print(f"Holdout accuracy: {acc:.3f}  |  AUC: {auc:.3f}  |  n_train={len(X_train)}  n_test={len(X_test)}")
    return clf, acc, auc
 
 
def train_aip_classifier_with_holdout(train_data, test_data, seed=0):
    """Train on a fixed training set and evaluate on a separate, pre-defined
    test set (e.g. the official train/test split shipped with a benchmark
    dataset like DeepAIP). More rigorous than an internal random split,
    since the test set was never seen during training or model selection.
    Returns (model, test_accuracy, test_auc).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
 
    X_train = np.array([featurize(s) for s, _ in train_data])
    y_train = np.array([label for _, label in train_data])
    X_test = np.array([featurize(s) for s, _ in test_data])
    y_test = np.array([label for _, label in test_data])
 
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
 
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
 
    print(f"Held-out test accuracy: {acc:.3f}  |  AUC: {auc:.3f}  |  n_train={len(X_train)}  n_test={len(X_test)}")
    return clf, acc, auc
 
 
def cross_validate(labeled_data, n_folds=5, seed=0):
    """Optional: k-fold CV for a more robust performance estimate than a
    single holdout split, useful since AIP datasets tend to be small."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
 
    X = np.array([featurize(s) for s, _ in labeled_data])
    y = np.array([label for _, label in labeled_data])
 
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"{n_folds}-fold CV AUC: mean={scores.mean():.3f}  std={scores.std():.3f}")
    return scores
 
 
# ---------------------------------------------------------------------------
# 3. Scoring generated candidates
# ---------------------------------------------------------------------------
 
def rank_candidates_by_aip_activity(candidates, clf, top_k=None):
    """Score generated sequences with the trained AIP classifier, sorted by
    predicted probability of anti-inflammatory activity (descending)."""
    results = []
    for seq in candidates:
        prob = clf.predict_proba([featurize(seq)])[0, 1]
        results.append({"sequence": seq, "predicted_aip_prob": round(float(prob), 3)})
    results.sort(key=lambda d: d["predicted_aip_prob"], reverse=True)
    return results[:top_k] if top_k else results