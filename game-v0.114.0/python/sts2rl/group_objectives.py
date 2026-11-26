"""Group objectives adapted from DeepSeekMath, DAPO and DSPO to game decisions.

An episode is a sequence, and an action chosen by the agent is one token.
Environment transitions are excluded from the likelihood.
"""
from __future__ import annotations

import numpy as np
import torch


def relative_advantages(rewards, epsilon=1e-8):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A group needs at least two finite rewards")
    std = values.std(ddof=0)
    if std <= epsilon:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / (std + epsilon)).astype(np.float32)


def action_weights(lengths, algorithm):
    """Weights sum to one; GRPO balances episodes, DAPO balances decisions."""
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Cannot optimize empty episodes")
    if algorithm == "dapo":
        return np.full(lengths.sum(), 1.0 / lengths.sum(), dtype=np.float32)
    if algorithm != "grpo":
        raise ValueError("Action weights apply to GRPO or DAPO")
    return np.repeat(1.0 / (len(lengths) * lengths), lengths).astype(np.float32)


def clipped_terms(log_ratio, advantage, clip_low, clip_high):
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    return torch.minimum(ratio * advantage, clipped * advantage)


def reference_kl(logp, reference_logp):
    """Nonnegative sampled k3 estimator from DeepSeekMath, equation 4."""
    delta = reference_logp - logp
    return torch.expm1(delta) - delta


def sequence_loss(mean_log_ratios, advantages, clip_low, clip_high):
    """DSPO equation 11: clip geometric-mean sequence ratios once per episode."""
    return -clipped_terms(mean_log_ratios, advantages, clip_low, clip_high).mean()


def sequence_coefficients(mean_log_ratios, advantages, clip_low, clip_high):
    """Exact d loss / d mean(log ratio), for memory-bounded two-pass gradients."""
    delta = mean_log_ratios.detach().requires_grad_(True)
    loss = sequence_loss(delta, advantages, clip_low, clip_high)
    coefficient, = torch.autograd.grad(loss, delta)
    return loss.detach(), coefficient.detach()


def overlong_penalty(lengths, maximum, buffer):
    lengths = np.asarray(lengths, dtype=np.float64)
    if maximum <= 0 or not 0 < buffer <= maximum:
        raise ValueError("Invalid overlong window")
    return -np.clip((lengths - (maximum - buffer)) / buffer, 0.0, 1.0)
