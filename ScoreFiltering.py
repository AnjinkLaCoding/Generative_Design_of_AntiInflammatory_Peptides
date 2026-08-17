from Utils import (
    net_charge,
    mean_hydrophobicity,
    hydrophobic_moment,
)


def max_identity_to_training_set(seq, training_seqs):
    """Cheap sequence-identity check (no alignment) to flag near-duplicates
    of the training data, i.e. memorized rather than novel sequences."""
    best = 0.0
    for t in training_seqs:
        L = min(len(seq), len(t))
        if L == 0:
            continue
        matches = sum(1 for a, b in zip(seq[:L], t[:L]))
        # also check the reverse alignment offset-free via simple overlap ratio
        common = len(set(zip(range(L), seq[:L])) & set(zip(range(L), t[:L])))
        identity = common / L
        best = max(best, identity)
    return best


def score_and_filter(
    candidates,
    training_seqs,
    charge_range=(2, 9),
    max_identity=0.9,
    top_k=None,
):
    """Filter generated candidates by basic AMP-like physicochemical rules
    and rank by a simple composite score. Returns a list of dicts.

    This is a heuristic gate, not a validated activity predictor -- treat
    it as a way to discard obvious junk, not as a claim of real activity.
    """
    training_set = set(training_seqs)
    scored = []
    for seq in set(candidates):
        if seq in training_set:
            continue  # exact duplicate of known AMP, not novel
        charge = net_charge(seq)
        hydro = mean_hydrophobicity(seq)
        moment = hydrophobic_moment(seq)
        identity = max_identity_to_training_set(seq, training_seqs)

        passes_charge = charge_range[0] <= charge <= charge_range[1]
        passes_novelty = identity <= max_identity

        # composite heuristic score: reward cationic + amphipathic peptides
        composite = charge * 0.4 + moment * 2.0 - max(0, hydro) * 0.1

        scored.append({
            "sequence": seq,
            "length": len(seq),
            "net_charge": round(charge, 2),
            "mean_hydrophobicity": round(hydro, 2),
            "hydrophobic_moment": round(moment, 2),
            "max_identity_to_known": round(identity, 2),
            "passes_filters": passes_charge and passes_novelty,
            "score": round(composite, 3),
        })

    scored.sort(key=lambda d: d["score"], reverse=True)
    passing = [d for d in scored if d["passes_filters"]]
    return passing[:top_k] if top_k else passing