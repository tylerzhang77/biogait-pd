"""Compatibility helpers for the released BioGait-PD checkpoint verifier.

The historical ablation driver depended on the private training workspace and on
optional modelling packages. The public verifier only needs the frozen fusion
configuration and the probability-combination function, so those pieces are
implemented here without importing the private training pipeline.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-9


# This is the configuration used to create the released reconstructed
# checkpoints and archived predictions. ``beta`` is the linear-correction
# weight, so the pooled EBM weight is 1 - beta = 0.8.
FINAL = {
    "beta": 0.2,
    "source_weight_scheme": "similarity_only",
    "kappa": 25.0,
    "tau": 1.0,
    "reliability_power": 1.0,
    "uniform_blend": 0.25,
    "fusion_mode": "arithmetic",
    "temperature": 0.8,
    "alpha_mode": "source_when_uncertain",
    "alpha_low": 0.6,
    "alpha_high": 0.95,
    "threshold_01": 0.65,
    "threshold_12": 1.45,
}


def normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPS, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def weighted_probability_sum(probability_by_source: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    result = None
    total = 0.0
    for source, probabilities in probability_by_source.items():
        weight = float(weights[source])
        result = weight * probabilities if result is None else result + weight * probabilities
        total += weight
    if result is None or total <= 0:
        raise ValueError("At least one source probability array and a positive total weight are required")
    return normalize_rows(result / total)


def source_ebm_shrunk(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    counts: dict[str, int],
    kappa: float,
) -> np.ndarray:
    source_probabilities = {}
    for source, roles in source_roles.items():
        shrinkage = counts[source] / (counts[source] + kappa)
        source_probabilities[source] = (
            shrinkage * roles["nonlinear_ebm"]
            + (1.0 - shrinkage) * population_roles["nonlinear_ebm"]
        )
    return weighted_probability_sum(source_probabilities, weights)


def fuse(population: np.ndarray, source: np.ndarray, alpha: float, temperature: float) -> np.ndarray:
    fused = (1.0 - alpha) * population + alpha * source
    fused = normalize_rows(fused)
    if abs(temperature - 1.0) > 1e-8:
        fused = normalize_rows(np.power(fused, 1.0 / temperature))
    return fused


def build_probabilities(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    counts: dict[str, int],
    variant: str,
    alpha: float,
) -> tuple[np.ndarray, str]:
    """Reconstruct final released probabilities from saved role outputs."""
    if variant != "final_similarity_gated_biogait_moe":
        raise ValueError(f"Unsupported public verification variant: {variant}")
    beta = FINAL["beta"]
    population = normalize_rows(
        (1.0 - beta) * population_roles["nonlinear_ebm"]
        + beta * population_roles["linear_margin_correction"]
    )
    source = source_ebm_shrunk(
        population_roles,
        source_roles,
        weights,
        counts,
        FINAL["kappa"],
    )
    return fuse(population, source, alpha, FINAL["temperature"]), "ordinal_threshold"
