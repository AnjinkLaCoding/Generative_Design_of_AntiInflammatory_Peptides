# Generative Design of Novel Anti-Inflammatory Peptides Using Markov Chains and Variational Autoencoders
## Introduction
This project uses generative sequence modeling — a Markov chain and a variational autoencoder (VAE) — to design novel AIP candidates. The VAE is pretrained on the broad APD3/APD6 antimicrobial peptide corpus and fine-tuned on a small anti-inflammatory-specific subset, addressing the limited training data available. Generated candidates are filtered for novelty and physicochemical plausibility, then ranked by a classifier(Random FOrest) trained on the DeepAIP dataset to prioritize sequences most likely to be genuinely anti-inflammatory. The pipeline is intended as a hypothesis-generation tool to guide future experimental testing.

## Datasets
- Test and Train dataset (Positive & Negative) for the classifier(RF)
- All available AMPs from APD6
- 104 Anti-Inflammatory AMPs

## Methods
1. Data
   -Two sets of peptide sequences were used: (1) the full APD3/APD6 antimicrobial peptide database, for broad sequence pretraining, and (2) a small subset (~104 sequences) annotated as anti-inflammatory within APD3/APD6, the actual design target. A separate, independently labeled dataset (DeepAIP: 4,480 peptides labeled AIP/non-AIP) was used for scoring.

3. Generation
   -Two generative models were trained to produce candidate sequences:
A Markov chain (order 1), trained directly on the anti-inflammatory subset, capturing local amino-acid transition patterns.
A VAE (LSTM encoder-decoder), first pretrained on the full AMP corpus to learn general peptide sequence structure, then fine-tuned on the small anti-inflammatory subset — a transfer-learning approach chosen specifically to compensate for the limited target-class data.

3. Filtering
   -Generated candidates were filtered to remove sequences that were near-duplicates of the training set (sequence identity ≥ 0.7, to avoid simple memorization) and those falling outside plausible physicochemical bounds (net charge, hydrophobicity).

5. Scoring
   -Surviving candidates were ranked using a Random Forest classifier trained on physicochemical and compositional features (charge, hydrophobicity, hydrophobic moment, amino acid composition) from the DeepAIP dataset, using its predefined train/test split for evaluation. This produced a predicted probability of anti-inflammatory activity for each candidate, used to rank the final shortlist.
