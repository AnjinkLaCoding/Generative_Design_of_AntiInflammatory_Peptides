import random
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_fasta(path, min_len=2, max_len=100, return_ids=False, verbose=True):
    """Parse a FASTA file into a list of clean, valid peptide sequences.
        >Your search led to 3306 peptides
        >AP00001
        GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV

        >AP00002
        YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY

    Any line starting with ">" is treated as a header/delimiter

    Args:
        min_len, max_len: length filter. Default max_len=100 matches APD3's
            own registration cutoff for natural AMP
        return_ids: if True, returns a list of (id, sequence) tuples instead
            of just sequences, using the last ">..." header seen before each
            sequence as the id (e.g. "AP00001").
        verbose: print a short summary of how many sequences were kept vs.
            dropped, and why, so silent data loss doesn't go unnoticed.
    """
    raw_entries = []
    current_seq = []
    current_id = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    raw_entries.append((current_id, "".join(current_seq)))
                    current_seq = []
                current_id = line[1:].strip()
            else:
                current_seq.append(line.upper())
        if current_seq:
            raw_entries.append((current_id, "".join(current_seq)))

    kept = []
    seen = set()
    n_too_short_long = 0
    n_bad_chars = 0
    n_dupe = 0
    for pid, s in raw_entries:
        if not (min_len <= len(s) <= max_len):
            n_too_short_long += 1
            continue
        if not set(s).issubset(AA_SET):
            n_bad_chars += 1
            continue  # drop sequences with non-standard residues (X, B, etc.)
        if s in seen:
            n_dupe += 1
            continue
        seen.add(s)
        kept.append((pid, s))

    if verbose:
        print(
            f"load_fasta({path}): {len(kept)} kept, "
            f"{n_too_short_long} dropped (length outside [{min_len},{max_len}]), "
            f"{n_bad_chars} dropped (non-standard residues), "
            f"{n_dupe} dropped (duplicate)"
        )

    if return_ids:
        return kept
    return [s for _, s in kept]

def peek_lengths(path):
    """Quick length-distribution check on a FASTA file, useful for choosing
    a sensible max_len before calling load_fasta()."""
    lengths = [len(s) for s in load_fasta(path, min_len=1, max_len=10_000, verbose=False)]
    if not lengths:
        print("No valid sequences found.")
        return
    lengths.sort()
    n = len(lengths)
    print(f"n={n}  min={lengths[0]}  max={lengths[-1]}  median={lengths[n // 2]}")
    for cutoff in (30, 40, 50, 60, 80, 100):
        print(f"  count > {cutoff}: {sum(1 for l in lengths if l > cutoff)}")

# ---------------------------------------------------------------------------
# 4. Physicochemical scoring / filtering
# ---------------------------------------------------------------------------

# Simplified charge contributions at physiological pH
POSITIVE = set("KR")
NEGATIVE = set("DE")
HISTIDINE_PARTIAL = 0.1  # H is only slightly positive at pH 7.4

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def net_charge(seq):
    return (
        sum(1 for c in seq if c in POSITIVE)
        - sum(1 for c in seq if c in NEGATIVE)
        + seq.count("H") * HISTIDINE_PARTIAL
    )


def mean_hydrophobicity(seq):
    return np.mean([KD_SCALE[c] for c in seq])


def hydrophobic_moment(seq, angle_deg=100):
    """Eisenberg hydrophobic moment: measures amphipathicity assuming an
    alpha-helix (100 degrees per residue). Higher = more amphipathic,
    a common feature of membrane-active AMPs."""
    angle = np.radians(angle_deg)
    sin_sum = sum(KD_SCALE[c] * np.sin(i * angle) for i, c in enumerate(seq))
    cos_sum = sum(KD_SCALE[c] * np.cos(i * angle) for i, c in enumerate(seq))
    return np.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(seq)