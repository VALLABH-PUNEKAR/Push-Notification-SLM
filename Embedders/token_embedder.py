"""
Token Embedding Layer + Tied Output Head
==========================================

Component position in the model:

    token IDs  -->  TokenEmbedding  -->  [transformer blocks: GQA + RoPE + SwiGLU]  -->  LMHead (tied)  -->  logits

This file implements exactly two things, per the handoff doc:

  1. TokenEmbedding
        - A lookup table of shape (vocab_size, hidden_size).
        - Converts input token IDs (batch_size, seq_len) into dense
          vectors (batch_size, seq_len, hidden_size).
        - Carries NO positional information (that's RoPE's job, applied
          later inside the attention block — not here).
        - Initialized with small random values (normal, std=0.02),
          matching GPT-2 / Llama-style init conventions.

  2. LMHead
        - The final projection back to vocab-sized logits.
        - Reuses (ties) the exact same weight tensor as TokenEmbedding,
          rather than holding its own separate (vocab_size, hidden_size)
          matrix. This is enforced structurally: LMHead does not own a
          weight parameter at all, it takes a TokenEmbedding instance
          and indexes into its .weight directly via F.linear.

Defaults:
  vocab_size = 8000   (per the finalized BBPE tokenizer decision)
  hidden_size = 512   (placeholder only — the real architecture decision
                        on hidden_size/num_layers/num_heads/num_kv_groups
                        is still open per the handoff doc; change this
                        when that's locked in)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """
    Input-side token embedding lookup table.

    Shape contract:
        input:  token_ids, LongTensor, shape (batch_size, seq_len)
                values must be in [0, vocab_size - 1]
        output: FloatTensor, shape (batch_size, seq_len, hidden_size)

    This module holds the single (vocab_size, hidden_size) weight matrix
    that is later shared (tied) with LMHead for the output projection.
    """

    def __init__(self, vocab_size: int = 8000, hidden_size: int = 512, init_std: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # nn.Embedding internally is just a (vocab_size, hidden_size)
        # weight matrix with a lookup (gather) forward pass — no extra
        # computation beyond the index-based lookup itself.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        # Small random init (mean 0, std ~0.02) rather than zeros — zero
        # init would give every token an identical starting vector with
        # no gradient signal to differentiate them early in training.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)

    @property
    def weight(self) -> nn.Parameter:
        """
        Exposes the underlying (vocab_size, hidden_size) weight tensor
        directly. LMHead reads this property to tie its output
        projection to this exact same Parameter object (same memory,
        same gradients) rather than a copy.
        """
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len) LongTensor of token IDs
        returns:   (batch_size, seq_len, hidden_size) FloatTensor
        """
        return self.embedding(token_ids)


class LMHead(nn.Module):
    """
    Output-side projection: final hidden states -> vocab-sized logits.

    This module does NOT own its own (vocab_size, hidden_size) weight
    matrix. It is constructed from a TokenEmbedding instance and reuses
    that instance's weight tensor for the projection (transposed), which
    is the standard tied-embeddings setup used by GPT-2, Llama, Mistral,
    etc.

    Shape contract:
        input:  hidden_states, FloatTensor, shape (batch_size, seq_len, hidden_size)
        output: logits, FloatTensor, shape (batch_size, seq_len, vocab_size)
    """

    def __init__(self, token_embedding: TokenEmbedding, bias: bool = False):
        super().__init__()
        if not isinstance(token_embedding, TokenEmbedding):
            raise TypeError("LMHead requires a TokenEmbedding instance to tie weights to.")

        self.token_embedding = token_embedding
        self.vocab_size = token_embedding.vocab_size
        self.hidden_size = token_embedding.hidden_size

        # No separate weight Parameter is created here on purpose — the
        # weight used in forward() is token_embedding.weight, looked up
        # live each call. This guarantees the input embedding and output
        # projection can never silently drift apart (e.g. one optimizer
        # step updating one copy but not the other), since there is only
        # ever one underlying tensor.
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        returns:       (batch_size, seq_len, vocab_size) logits
        """
        # F.linear(x, W) computes x @ W^T. Using the embedding weight
        # directly here (shape vocab_size x hidden_size) reproduces the
        # standard tied-weight output projection without ever
        # materializing a second (vocab_size, hidden_size) matrix.
        return F.linear(hidden_states, self.token_embedding.weight, self.bias)


def build_tied_embedding_and_head(
    vocab_size: int = 8000,
    hidden_size: int = 512,
    init_std: float = 0.02,
    lm_head_bias: bool = False,
) -> tuple[TokenEmbedding, LMHead]:
    """
    Convenience constructor returning a (TokenEmbedding, LMHead) pair
    that share weights, since in practice these two modules are always
    built together in the parent model's __init__.
    """
    token_embedding = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size, init_std=init_std)
    lm_head = LMHead(token_embedding, bias=lm_head_bias)
    return token_embedding, lm_head