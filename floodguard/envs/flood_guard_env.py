"""Ambiente Gymnasium FloodGuard (versao simplificada).

Esqueleto criado na Etapa 0 (setup do projeto). A implementacao completa do
MDP (estado, acao, transicao, recompensa, terminacao) descrito na Secao 3 de
`proposta-validacao-enchente-simplificada.md` e responsabilidade da Etapa 1.
"""

import gymnasium as gym
from gymnasium import spaces


class FloodGuardEnv(gym.Env):
    """Robo autonomo resgata um animal ilhado antes que a enchente o alcance.

    Ver `proposta-validacao-enchente-simplificada.md` para a formalizacao
    completa do MDP. Esta classe ainda nao implementa a dinamica do
    ambiente (Etapa 1).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, grid_size: int = 6, render_mode: str | None = None):
        self.grid_size = grid_size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "robot_pos": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=int),
                "battery": spaces.Box(low=0, high=100, shape=(1,), dtype=int),
                "flood_map": spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=int),
                "barrier_map": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=int),
                "barrier_kits": spaces.Box(low=0, high=2, shape=(1,), dtype=int),
                "animal_status": spaces.Discrete(4),
                "animal_pos": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=int),
                "steps_left": spaces.Box(low=0, high=500, shape=(1,), dtype=int),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        raise NotImplementedError("Implementado na Etapa 1")

    def step(self, action):
        raise NotImplementedError("Implementado na Etapa 1")

    def render(self):
        raise NotImplementedError("Implementado na Etapa 2")
