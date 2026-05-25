"""P3SL research harness.

A clean, single-process re-implementation of the Personalized Privacy-Preserving
Split Learning protocol. The Docker-based code in
``my_implementation/P3SL_Implementation`` is the *production* reference; this
package is the *experimental* reference used to generate publication-grade
numbers (sweeps, baselines, multi-seed averages, Pareto curves).
"""

from .model import P3SLModel
from .simulator import P3SLSimulator, P3SLConfig, ClientProfile
from .dp import (
    LaplaceMechanism,
    GaussianMechanism,
    laplace_epsilon,
    gaussian_rdp_epsilon,
)
from .metrics import (
    PrivacyMetrics,
    compute_privacy_metrics,
    mia_auc,
    backdoor_asr,
)

__all__ = [
    "P3SLModel",
    "P3SLSimulator",
    "P3SLConfig",
    "ClientProfile",
    "LaplaceMechanism",
    "GaussianMechanism",
    "laplace_epsilon",
    "gaussian_rdp_epsilon",
    "PrivacyMetrics",
    "compute_privacy_metrics",
    "mia_auc",
    "backdoor_asr",
]
