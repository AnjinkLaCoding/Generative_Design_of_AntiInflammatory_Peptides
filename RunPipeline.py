import argparse
import csv
import sys
import pandas as pd
from AMPGenerator import (
    MarkovChainGenerator,
    VAEConfig,
    train_vae,
    finetune_vae,
    sample_vae,
)

from Utils import(load_fasta, peek_lengths)

from ScoreFiltering import score_and_filter

from AIPScoring import (
    load_labeled_fasta_pair,
    rank_candidates_by_aip_activity,
    train_aip_classifier_with_holdout,
)

# ===========================================================================
# CONFIG -- edit these paths and settings, then hit Run
# ===========================================================================
 
# --- Input files ---
# full AMP set (for VAE pretraining)
ALL_AMP_FASTA = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/naturalAMPs_APD2024a_all.fasta"
#104 anti-inflammatory-only AMPs
ANTIINFLAM_FASTA = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/AntiInflammation.fasta"

# AIP classifier training data
AIP_POSITIVE_TRAIN = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/Training_RF/pos_1365_train.fasta"
AIP_NEGATIVE_TRAIN = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/Training_RF/neg_2218_train.fasta"
# AIP classifier held-out test data
AIP_POSITIVE_TEST = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/Test_RF/pos_342_test.fasta"
AIP_NEGATIVE_TEST = "C:/Users/Matthew/Downloads/AMPProj1/AMPGenerator_AntiInflamm/Test_RF/neg_555_test.fasta"
 
# --- Output ---
OUTPUT_CSV = "ranked_candidates.csv"
 
# --- Generation settings ---
N_CANDIDATES = 300          # raw candidates to generate per model, before filtering
TOP_K = 50                  # how many top-ranked candidates to keep in the final output
MAX_IDENTITY = 0.7          # reject candidates this similar or more to a training sequence
MARKOV_ORDER = 1            # 1 is safer for small datasets like ~100 sequences
 
# --- VAE settings ---
SKIP_VAE = False            # set True to only run the Markov chain (e.g. if torch isn't installed)
VAE_PRETRAIN_EPOCHS = 100
VAE_FINETUNE_EPOCHS = 30
 
# ===========================================================================
# Pipeline (no need to edit below this line)
# ===========================================================================
 
 
def main():
    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading FASTA files")
    print("=" * 70)
    print("\n-- Anti-inflammatory subset (target) --")
    peek_lengths(ANTIINFLAM_FASTA)
    antiinflam_seqs = load_fasta(ANTIINFLAM_FASTA)
 
    print("\n-- Full AMP set (for VAE pretraining) --")
    all_amp_seqs = load_fasta(ALL_AMP_FASTA)
 
    if len(antiinflam_seqs) < 10:
        print(f"\nWARNING: only {len(antiinflam_seqs)} anti-inflammatory sequences loaded. "
              "Check ANTIINFLAM_FASTA path and max_len before continuing.")
        sys.exit(1)
 
    # -----------------------------------------------------------------
    # 2. Train generators
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Training generators")
    print("=" * 70)
 
    print(f"\n-- Markov chain (order={MARKOV_ORDER}), trained on anti-inflammatory subset only --")
    mc = MarkovChainGenerator(order=MARKOV_ORDER).fit(antiinflam_seqs)
    mc_candidates = mc.generate(n=N_CANDIDATES)
    print(f"Generated {len(mc_candidates)} raw candidates")
 
    vae_candidates = []
    if not SKIP_VAE:
        try:
            print(f"\n-- VAE: pretrain on {len(all_amp_seqs)} AMPs, fine-tune on {len(antiinflam_seqs)} anti-inflammatory --")
            cfg = VAEConfig(hidden_dim=48, latent_dim=8, epochs=VAE_PRETRAIN_EPOCHS)
            model = train_vae(all_amp_seqs, cfg)
            model = finetune_vae(model, antiinflam_seqs, epochs=VAE_FINETUNE_EPOCHS)
            vae_candidates = sample_vae(model, n=N_CANDIDATES)
            print(f"Generated {len(vae_candidates)} raw candidates")
        except ImportError:
            print("torch not installed -- skipping VAE. Run: pip install torch --break-system-packages")
 
    # -----------------------------------------------------------------
    # 3. Filter for novelty + physicochemical plausibility
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Filtering candidates")
    print("=" * 70)
 
    all_raw_candidates = list(set(mc_candidates + vae_candidates))
    print(f"\n{len(all_raw_candidates)} unique raw candidates pooled from both generators")
 
    filtered = score_and_filter(
        all_raw_candidates,
        training_seqs=antiinflam_seqs,
        max_identity=MAX_IDENTITY,
    )
    print(f"{len(filtered)} passed novelty + physicochemical filters")
 
    if not filtered:
        print("\nNo candidates survived filtering. Try relaxing MAX_IDENTITY or "
              "the charge_range in score_and_filter().")
        sys.exit(1)
 
    # -----------------------------------------------------------------
    # 4. Score with the anti-inflammatory activity classifier
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Scoring with the AIP activity classifier")
    print("=" * 70)
 
    print("\nTraining on the AIP train split, evaluating on the held-out AIP test split")
    train_data = load_labeled_fasta_pair(AIP_POSITIVE_TRAIN, AIP_NEGATIVE_TRAIN)
    test_data = load_labeled_fasta_pair(AIP_POSITIVE_TEST, AIP_NEGATIVE_TEST)
    clf, acc, auc = train_aip_classifier_with_holdout(train_data, test_data)
 
    filtered_seqs = [d["sequence"] for d in filtered]
    ranked = rank_candidates_by_aip_activity(filtered_seqs, clf, top_k=TOP_K)
 
    # merge in the physicochemical stats for the surviving top candidates
    filter_lookup = {d["sequence"]: d for d in filtered}
    for r in ranked:
        stats = filter_lookup[r["sequence"]]
        r.update({
            "length": stats["length"],
            "net_charge": stats["net_charge"],
            "hydrophobic_moment": stats["hydrophobic_moment"],
            "max_identity_to_known": stats["max_identity_to_known"],
        })

    df = pd.DataFrame(ranked)
    print(df.head(5))
 
    # -----------------------------------------------------------------
    # 5. Save results
    # -----------------------------------------------------------------
    #print("\n" + "=" * 70)
    #print(f"STEP 5: Top {len(ranked)} candidates (saved to {OUTPUT_CSV})")
    #print("=" * 70 + "\n")
 
    #fieldnames = ["sequence", "predicted_aip_prob", "length", "net_charge",
                  #"hydrophobic_moment", "max_identity_to_known"]
    #with open(OUTPUT_CSV, "w", newline="") as f:
        #writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer.writeheader()
        #writer.writerows(ranked)
 
    #for r in ranked[:15]:
        #print(r)
 
    #print(f"\nFull ranked list saved to: {OUTPUT_CSV}")
 
 
if __name__ == "__main__":
    main()


#The result of top 5 sequences
#                           sequence                     predicted_aip_prob  length  net_charge  hydrophobic_moment  max_identity_to_known
#0                                   IRILKLIPSLFHKSIM               0.883      16         3.1                1.39                   0.42
#1  GLSIFQIAAPEEADKVKGKRKKKSLRILTNLLKQCRRLLRALVGIY...               0.880      50         9.0                0.87                   0.23
#2                                GIFLKIALKIAKSLASLFL               0.870      19         3.0                1.18                   0.32
#3                                       ALKKFLRISLIN               0.870      12         3.0                0.74                   0.33
#4              KIFIKTRLQKLLSTLSNAAVKASVTREGAAKHLIQEL               0.863      37         5.1                0.81                   0.33