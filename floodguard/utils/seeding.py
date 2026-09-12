"""Utilitario de reprodutibilidade compartilhado por todos os experimentos.

Usar `set_global_seed` no inicio de qualquer script de treino/avaliacao
(Etapas 3-6) para que a mesma seed apareca registrada em todos os
resultados reportados no relatorio.
"""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Fixa a seed do Python, NumPy e (se instalado) PyTorch. Retorna a seed usada."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
