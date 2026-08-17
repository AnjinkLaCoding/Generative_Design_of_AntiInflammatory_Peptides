# Generative Design of Novel Anti-Inflammatory Peptides Using Markov Chains and Variational Autoencoders
## Introduction
This project uses generative sequence modeling — a Markov chain and a variational autoencoder (VAE) — to design novel AIP candidates. The VAE is pretrained on the broad APD3/APD6 antimicrobial peptide corpus and fine-tuned on a small anti-inflammatory-specific subset, addressing the limited training data available. Generated candidates are filtered for novelty and physicochemical plausibility, then ranked by a classifier(Random FOrest) trained on the DeepAIP dataset to prioritize sequences most likely to be genuinely anti-inflammatory. The pipeline is intended as a hypothesis-generation tool to guide future experimental testing.

##Dataset
