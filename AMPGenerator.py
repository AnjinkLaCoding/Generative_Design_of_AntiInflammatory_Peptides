"""
AMP Generative Design Framework
================================

A minimal, extensible pipeline for training generative models on known
antimicrobial peptide (AMP) sequences and generating novel candidates.

Includes:
    1. Data loading/preprocessing from FASTA
    2. A Markov chain baseline generator (fast, no deep learning needed)
    3. An LSTM-VAE generator (PyTorch) for a learned latent space
    4. Physicochemical filtering/scoring of generated candidates

Usage sketch:
    seqs = load_fasta("amps.fasta")
    mc = MarkovChainGenerator(order=2).fit(seqs)
    candidates = mc.generate(n=200)

    vae = train_vae(seqs, epochs=50)
    candidates = sample_vae(vae, n=200, seq_len=30)

    scored = score_and_filter(candidates, training_seqs=seqs)
"""

import random
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
from Utils import AMINO_ACIDS


# ---------------------------------------------------------------------------
# 2. Markov chain baseline
# ---------------------------------------------------------------------------

class MarkovChainGenerator:
    """Order-N Markov chain over amino acid sequences.

    Simple, fast, and a good sanity-check baseline before investing in a
    neural model. Captures local composition/transition biases but not
    long-range structure (e.g. amphipathic helical periodicity).
    """

    START = "^"
    END = "$"

    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.length_dist = []

    def fit(self, sequences):
        for seq in sequences:
            padded = self.START * self.order + seq + self.END
            self.length_dist.append(len(seq))
            for i in range(len(padded) - self.order):
                context = padded[i:i + self.order]
                nxt = padded[i + self.order]
                self.transitions[context][nxt] += 1
        return self

    def _sample_length(self):
        return random.choice(self.length_dist)

    def generate(self, n=100, max_len=50):
        out = []
        attempts = 0
        while len(out) < n and attempts < n * 20:
            attempts += 1
            context = self.START * self.order
            seq = []
            for _ in range(max_len):
                choices = self.transitions.get(context)
                if not choices:
                    break
                nxt = random.choices(
                    list(choices.keys()), weights=list(choices.values())
                )[0]
                if nxt == self.END:
                    break
                seq.append(nxt)
                context = (context + nxt)[-self.order:]
            candidate = "".join(seq)
            if 8 <= len(candidate) <= max_len:
                out.append(candidate)
        return out


# ---------------------------------------------------------------------------
# 3. LSTM-VAE generator (PyTorch)
# ---------------------------------------------------------------------------
# Requires: pip install torch --break-system-packages

def _lazy_import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


@dataclass
class VAEConfig:
    vocab_size: int = len(AMINO_ACIDS) + 2  # + PAD + EOS
    embed_dim: int = 32
    hidden_dim: int = 128
    latent_dim: int = 16
    max_len: int = 40
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3


def _build_vocab():
    # 0 = PAD, 1 = EOS, then amino acids
    stoi = {"<PAD>": 0, "<EOS>": 1}
    for i, aa in enumerate(AMINO_ACIDS):
        stoi[aa] = i + 2
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


def encode_sequences(seqs, stoi, max_len):
    import torch
    batch = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids = [stoi[c] for c in s[: max_len - 1]] + [stoi["<EOS>"]]
        ids = ids[:max_len]
        batch[i, : len(ids)] = torch.tensor(ids)
    return batch


class LSTMVAE:
    """Encoder-decoder LSTM VAE over amino acid sequences.

    The latent space is the interesting part: once trained, you can sample
    randomly (novel generation), interpolate between two known AMPs'
    encodings, or do arithmetic in latent space (e.g. "more cationic"
    direction) if you additionally regress physicochemical properties
    against latent coordinates.
    """

    def __init__(self, config: VAEConfig):
        torch, nn, F = _lazy_import_torch()
        self.torch, self.nn, self.F = torch, nn, F
        self.cfg = config
        self.stoi, self.itos = _build_vocab()

        cfg = config
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder_rnn = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        self.params = (
            list(self.embed.parameters())
            + list(self.encoder_rnn.parameters())
            + list(self.fc_mu.parameters())
            + list(self.fc_logvar.parameters())
            + list(self.latent_to_hidden.parameters())
            + list(self.decoder_rnn.parameters())
            + list(self.output_proj.parameters())
        )

    def encode(self, x):
        emb = self.embed(x)
        _, (h, _) = self.encoder_rnn(emb)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = self.torch.exp(0.5 * logvar)
        eps = self.torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_input):
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = self.torch.zeros_like(h0)
        emb = self.embed(x_input)
        out, _ = self.decoder_rnn(emb, (h0, c0))
        return self.output_proj(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # teacher forcing: shift input right, decoder predicts next token
        decoder_input = self.torch.cat(
            [self.torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1
        )
        logits = self.decode(z, decoder_input)
        return logits, mu, logvar

    def loss_fn(self, logits, target, mu, logvar, kl_weight=0.5):
        F = self.F
        recon = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=0,  # PAD
        )
        kl = -0.5 * self.torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl_weight * kl, recon.item(), kl.item()

    @property
    def device_params(self):
        return self.params


def train_vae(sequences, config: VAEConfig = None, verbose=True):
    """Train an LSTM-VAE on a list of AMP sequences. Returns the trained model."""
    torch, nn, F = _lazy_import_torch()
    cfg = config or VAEConfig()
    model = LSTMVAE(cfg)
    data = encode_sequences(sequences, model.stoi, cfg.max_len)

    optimizer = torch.optim.Adam(model.device_params, lr=cfg.lr)
    n = data.size(0)

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            batch = data[idx]
            logits, mu, logvar = model.forward(batch)
            # anneal KL weight so the model doesn't ignore the latent code early on
            kl_weight = min(0.5, epoch / max(cfg.epochs * 0.5, 1) * 0.5)
            loss, recon, kl = model.loss_fn(logits, batch, mu, logvar, kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{cfg.epochs}  loss={total_loss/n:.4f}")
    return model


def finetune_vae(model: LSTMVAE, sequences, epochs=30, lr=5e-4, verbose=True):
    """Continue training an already-trained VAE on a smaller, specialized
    set of sequences (e.g. pretrain on all of APD3, then fine-tune on a
    small anti-inflammatory subset). Uses a lower learning rate and fewer
    epochs than train_vae by default, since the goal is to nudge the model
    toward the new distribution without catastrophically overfitting to a
    small dataset.
    """
    torch = model.torch
    cfg = model.cfg
    data = encode_sequences(sequences, model.stoi, cfg.max_len)

    optimizer = torch.optim.Adam(model.device_params, lr=lr)
    n = data.size(0)
    batch_size = min(cfg.batch_size, n)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = data[idx]
            logits, mu, logvar = model.forward(batch)
            # keep KL weight fixed (already past the annealing phase from pretraining)
            loss, recon, kl = model.loss_fn(logits, batch, mu, logvar, kl_weight=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"finetune epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}")
    return model


def sample_vae(model: LSTMVAE, n=100, temperature=1.0):
    """Sample n novel sequences by drawing z ~ N(0, I) and decoding autoregressively."""
    torch = model.torch
    cfg = model.cfg
    itos = model.itos
    results = []

    with torch.no_grad():
        for _ in range(n):
            z = torch.randn(1, cfg.latent_dim)
            h = model.latent_to_hidden(z).unsqueeze(0)
            c = torch.zeros_like(h)
            token = torch.zeros(1, 1, dtype=torch.long)  # start token = PAD id, used as BOS here
            out_chars = []
            for _ in range(cfg.max_len):
                emb = model.embed(token)
                out, (h, c) = model.decoder_rnn(emb, (h, c))
                logits = model.output_proj(out).squeeze(1) / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                if next_id == model.stoi["<EOS>"]:
                    break
                if next_id != 0:
                    out_chars.append(itos[next_id])
                token = torch.tensor([[next_id]])
            seq = "".join(out_chars)
            if len(seq) >= 6:
                results.append(seq)
    return results