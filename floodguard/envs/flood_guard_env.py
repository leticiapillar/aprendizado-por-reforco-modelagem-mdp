"""Ambiente Gymnasium FloodGuard (versao simplificada).

Implementa o MDP descrito na Secao 3 de
`proposta-validacao-enchente-simplificada.md`: um robo autonomo precisa
resgatar um unico animal ilhado antes que a agua o alcance, administrando
bateria limitada e podendo instalar barreiras para conter o avanco da
enchente, em um grid 6x6 totalmente observavel.

Decisoes de implementacao nao fixadas na proposta (valores exatos de
recompensa, custos de bateria, probabilidades `p`/`q`, posicoes fixas do
cenario, numero maximo de steps) ficam expostas como parametros do
construtor com valores-padrao razoaveis; a calibracao fina acontece na
Etapa 3 (baseline + ajuste do MDP), sem exigir mudanca de codigo.
"""

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FloodGuardEnv(gym.Env):
    """Robo autonomo resgata um animal ilhado antes que a enchente o alcance.

    Ver `proposta-validacao-enchente-simplificada.md` para a formalizacao
    completa do MDP (estado, acao, transicao, recompensa, terminacao).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    # Niveis de inundacao de uma celula.
    DRY, SHALLOW, DEEP = 0, 1, 2

    # Status do animal.
    AWAITING, WITH_ROBOT, SAVED, LOST = 0, 1, 2, 3

    # Acoes.
    UP, DOWN, LEFT, RIGHT, INSTALL_BARRIER, WAIT_CHARGE = range(6)

    _ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    _NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        grid_size: int = 6,
        max_steps: int = 100,
        max_battery: int = 100,
        initial_barrier_kits: int = 2,
        flood_source: tuple[int, int] = (0, 0),
        robot_base: tuple[int, int] = (5, 5),
        animal_start: tuple[int, int] = (0, 5),
        safe_zone: tuple[int, int] = (5, 0),
        move_cost_dry: int = 1,
        move_cost_shallow: int = 3,
        barrier_install_cost: int = 15,
        charge_rate: int = 20,
        flood_advance_prob: float = 0.15,
        flood_deepen_prob: float = 0.10,
        rescue_reward: float = 20.0,
        barrier_effective_reward: float = 5.0,
        step_penalty: float = -1.0,
        animal_lost_penalty: float = -20.0,
        battery_depleted_penalty: float = -30.0,
        render_mode: Optional[str] = None,
    ):
        for name, pos in (
            ("flood_source", flood_source),
            ("robot_base", robot_base),
            ("animal_start", animal_start),
            ("safe_zone", safe_zone),
        ):
            if not (0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size):
                raise ValueError(f"{name}={pos} fora do grid {grid_size}x{grid_size}")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.max_battery = max_battery
        self.initial_barrier_kits = initial_barrier_kits
        self.flood_source = tuple(flood_source)
        self.robot_base = tuple(robot_base)
        self.animal_start = tuple(animal_start)
        self.safe_zone = tuple(safe_zone)
        self.move_cost_dry = move_cost_dry
        self.move_cost_shallow = move_cost_shallow
        self.barrier_install_cost = barrier_install_cost
        self.charge_rate = charge_rate
        self.flood_advance_prob = flood_advance_prob
        self.flood_deepen_prob = flood_deepen_prob
        self.rescue_reward = rescue_reward
        self.barrier_effective_reward = barrier_effective_reward
        self.step_penalty = step_penalty
        self.animal_lost_penalty = animal_lost_penalty
        self.battery_depleted_penalty = battery_depleted_penalty
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "robot_pos": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                "battery": spaces.Box(low=0, high=max_battery, shape=(1,), dtype=np.int32),
                "flood_map": spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.int32),
                "barrier_map": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int32),
                "barrier_kits": spaces.Box(low=0, high=initial_barrier_kits, shape=(1,), dtype=np.int32),
                "animal_status": spaces.Discrete(4),
                "animal_pos": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                "steps_left": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
            }
        )

        # Inicializados de fato em reset(); declarados aqui por clareza.
        self._steps = 0
        self.battery = self.max_battery
        self.robot_pos = np.array(self.robot_base, dtype=np.int32)
        self._facing = (0, 1)
        self.kits_remaining = self.initial_barrier_kits
        self.flood = np.full((grid_size, grid_size), self.DRY, dtype=np.int32)
        self.barriers = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.animal_status = self.AWAITING
        self.animal_pos = np.array(self.animal_start, dtype=np.int32)

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._steps = 0
        self.battery = self.max_battery
        self.robot_pos = np.array(self.robot_base, dtype=np.int32)
        self._facing = (0, 1)  # apenas para renderizacao (Etapa 2); nao faz parte do estado do MDP
        self.kits_remaining = self.initial_barrier_kits
        self.barriers = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.animal_status = self.AWAITING
        self.animal_pos = np.array(self.animal_start, dtype=np.int32)

        self.flood = np.full((self.grid_size, self.grid_size), self.DRY, dtype=np.int32)
        self.flood[self.flood_source] = self.SHALLOW

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Acao invalida: {action}")

        reward = self.step_penalty
        terminated = False

        if action in self._ACTION_DELTAS:
            reward += self._handle_move(action)
        elif action == self.INSTALL_BARRIER:
            reward += self._handle_install_barrier()
        elif action == self.WAIT_CHARGE:
            self._handle_wait_charge()

        self.battery = int(np.clip(self.battery, 0, self.max_battery))

        # Resgate automatico ao ocupar a celula do animal; entrega automatica
        # (com recompensa) ao ocupar a zona segura carregando o animal.
        robot_cell = tuple(self.robot_pos.tolist())
        if self.animal_status == self.AWAITING and robot_cell == self.animal_start:
            self.animal_status = self.WITH_ROBOT
        if self.animal_status == self.WITH_ROBOT and robot_cell == self.safe_zone:
            self.animal_status = self.SAVED
            reward += self.rescue_reward
            terminated = True

        if not terminated:
            self._advance_flood()

            animal_cell = tuple(self.animal_pos.tolist())
            if self.animal_status == self.AWAITING and self.flood[animal_cell] == self.DEEP:
                self.animal_status = self.LOST
                reward += self.animal_lost_penalty
                terminated = True

        if not terminated and self.battery <= 0 and robot_cell != self.robot_base:
            reward += self.battery_depleted_penalty
            terminated = True

        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "Chame gym.make(..., render_mode='rgb_array') (ou instancie "
                "FloodGuardEnv(render_mode='rgb_array')) antes de chamar render()."
            )
            return None
        if self.render_mode == "rgb_array":
            from floodguard.rendering.renderer import render_frame

            return render_frame(self)
        raise NotImplementedError(f"render_mode={self.render_mode!r} nao suportado")

    # ------------------------------------------------------------------
    # Dinamica interna
    # ------------------------------------------------------------------

    def _handle_move(self, action: int) -> float:
        dr, dc = self._ACTION_DELTAS[action]
        r, c = int(self.robot_pos[0]) + dr, int(self.robot_pos[1]) + dc
        self._facing = (dr, dc)

        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return 0.0  # bateu na borda do grid: nao se move, sem custo extra
        if self.flood[r, c] == self.DEEP:
            return 0.0  # agua profunda e intransponivel: nao se move

        cost = self.move_cost_dry if self.flood[r, c] == self.DRY else self.move_cost_shallow
        self.robot_pos = np.array([r, c], dtype=np.int32)
        self.battery -= cost
        return 0.0

    def _handle_install_barrier(self) -> float:
        cell = tuple(self.robot_pos.tolist())
        if (
            self.kits_remaining <= 0
            or self.barriers[cell]
            or self.flood[cell] == self.DEEP
            or self.battery < self.barrier_install_cost
        ):
            return 0.0  # acao invalida nesta celula/estado: no-op

        # Proxy computavel para "protege uma celula que seria inundada nos
        # steps seguintes": a celula ja estava em risco imediato de avancar
        # (seca e adjacente a agua, ou rasa prestes a aprofundar).
        effective = (
            self.flood[cell] == self.SHALLOW
            or (self.flood[cell] == self.DRY and self._is_adjacent_to_flooded(cell))
        )

        self.barriers[cell] = 1
        self.kits_remaining -= 1
        self.battery -= self.barrier_install_cost
        return self.barrier_effective_reward if effective else 0.0

    def _handle_wait_charge(self) -> None:
        if tuple(self.robot_pos.tolist()) == self.robot_base:
            self.battery += self.charge_rate

    def _is_adjacent_to_flooded(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        for dr, dc in self._NEIGHBOR_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                if self.flood[rr, cc] in (self.SHALLOW, self.DEEP):
                    return True
        return False

    def _advance_flood(self) -> None:
        """Avanco estocastico da agua (Secao 3 da proposta).

        Cada celula seca adjacente a uma celula ja inundada tem
        probabilidade `flood_advance_prob` de virar agua rasa; cada celula
        de agua rasa tem probabilidade `flood_deepen_prob` de virar agua
        profunda. Celulas com barreira instalada nao sofrem transicao
        (probabilidade de inundacao reduzida a ~zero). As transicoes usam o
        estado no INICIO do step (nao ha propagacao em cascata dentro do
        mesmo step).
        """
        new_flood = self.flood.copy()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.barriers[r, c]:
                    continue
                level = self.flood[r, c]
                if level == self.DRY and self._is_adjacent_to_flooded((r, c)):
                    if self.np_random.random() < self.flood_advance_prob:
                        new_flood[r, c] = self.SHALLOW
                elif level == self.SHALLOW:
                    if self.np_random.random() < self.flood_deepen_prob:
                        new_flood[r, c] = self.DEEP
        self.flood = new_flood

    # ------------------------------------------------------------------
    # Observacao / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        return {
            "robot_pos": self.robot_pos.astype(np.int32),
            "battery": np.array([self.battery], dtype=np.int32),
            "flood_map": self.flood.astype(np.int32),
            "barrier_map": self.barriers.astype(np.int32),
            "barrier_kits": np.array([self.kits_remaining], dtype=np.int32),
            "animal_status": int(self.animal_status),
            "animal_pos": self.animal_pos.astype(np.int32),
            "steps_left": np.array([max(0, self.max_steps - self._steps)], dtype=np.int32),
        }

    def _get_info(self) -> dict:
        return {
            "steps": self._steps,
            "battery": self.battery,
            "kits_remaining": self.kits_remaining,
            "animal_status": self.animal_status,
            "is_success": self.animal_status == self.SAVED,
        }
