"""Smoke test da Etapa 0: confirma que o ambiente de desenvolvimento esta
utilizavel antes de iniciar a implementacao do MDP (Etapa 1).
"""

import gymnasium
import optuna
import stable_baselines3

import floodguard
from floodguard.utils import set_global_seed


def test_dependencies_importable():
    assert gymnasium.__version__
    assert stable_baselines3.__version__
    assert optuna.__version__


def test_env_is_registered():
    assert "FloodGuard-v0" in gymnasium.envs.registry


def test_set_global_seed_is_deterministic():
    import random

    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b
