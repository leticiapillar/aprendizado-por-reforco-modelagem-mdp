"""Treino default de DQN, PPO e A2C para a Etapa 4.

Este script valida a integracao entre o FloodGuardEnv e o stable-baselines3
sem otimizar hiperparametros ainda. A ideia e manter todos os algoritmos com
os defaults do SB3 e variar apenas o algoritmo escolhido na linha de comando.

Exemplos:
    python experiments/train.py --algo dqn
    python experiments/train.py --algo ppo --total-timesteps 20000
    python experiments/train.py --algo a2c --eval-episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from floodguard.envs.flood_guard_env import FloodGuardEnv
from floodguard.utils.seeding import DEFAULT_SEED, set_global_seed

try:
    from experiments.random_baseline import (
        ANIMAL_STATUS_LABELS,
        EpisodeMetrics,
        evaluate_random_policy,
        summarize_metrics,
    )
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    # Quando o arquivo e chamado como `python experiments/train.py`, Python
    # coloca `experiments/` no sys.path. O fallback abaixo preserva esse uso
    # direto sem quebrar os imports por pacote usados nos testes.
    from random_baseline import (  # type: ignore[no-redef]
        ANIMAL_STATUS_LABELS,
        EpisodeMetrics,
        evaluate_random_policy,
        summarize_metrics,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = PROJECT_ROOT / "results" / "logs"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "results" / "models"

# O plano da Etapa 4 pede exatamente estes tres algoritmos. Guardar as classes
# em um dicionario evita condicionais repetidos e deixa a CLI alinhada ao SB3.
ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

# O ambiente original usa observacao Dict. Como queremos testar os tres
# algoritmos com a mesma politica default, achatamos a observacao e usamos
# MlpPolicy em todos os casos.
DEFAULT_POLICY = "MlpPolicy"

# Campos extras do info que o Monitor deve gravar no CSV de episodios. Eles
# reaproveitam as metricas fixadas na Etapa 3 para facilitar a comparacao.
MONITOR_INFO_KEYWORDS = (
    "is_success",
    "termination_reason",
    "battery_spent",
    "barriers_installed",
    "effective_barriers_installed",
    "pickup_step",
    "delivery_step",
)


class RescueProgressReward(gym.Wrapper):
    """Adiciona sinal denso de progresso sem alterar o objetivo do resgate.

    Antes da coleta, a distancia considera o caminho robo -> animal -> zona
    segura. Depois da coleta, considera apenas robo -> zona segura. Portanto,
    aproximar-se do proximo objetivo vale +5 por celula e afastar-se vale -5.
    A recompensa original de entrega continua sendo o incentivo principal.
    """

    def __init__(self, env: gym.Env, scale: float = 5.0):
        super().__init__(env)
        self.scale = scale
        self._previous_distance = 0

    @staticmethod
    def _manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _remaining_distance(self) -> int:
        base_env: FloodGuardEnv = self.env.unwrapped
        robot = tuple(int(value) for value in base_env.robot_pos)
        if base_env.animal_status == FloodGuardEnv.AWAITING:
            return self._manhattan(robot, base_env.animal_start) + self._manhattan(
                base_env.animal_start, base_env.safe_zone
            )
        if base_env.animal_status == FloodGuardEnv.WITH_ROBOT:
            return self._manhattan(robot, base_env.safe_zone)
        return 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._previous_distance = self._remaining_distance()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_distance = self._remaining_distance()

        # Em finais malsucedidos nao concedemos artificialmente o progresso
        # restante. O bonus terminal deve existir somente na entrega real.
        if terminated and not info["is_success"]:
            progress = 0
        else:
            progress = self._previous_distance - current_distance

        self._previous_distance = current_distance
        shaped_progress = self.scale * progress
        info["progress_reward"] = float(shaped_progress)
        return observation, float(reward + shaped_progress), terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    """Normaliza a observacao achatada para float32 em [0, 1].

    O `Box` achatado do ambiente e inteiro (int64). O `RescaleObservation` do
    Gymnasium preserva esse dtype e trunca os valores para inteiros, o que
    zerava posicao, bateria e passos restantes. Aqui a conversao para float32
    ocorre antes da divisao pela amplitude do espaco (eixos de amplitude zero,
    como os kits com `initial_barrier_kits=0`, ficam em 0).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        self._low = space.low.astype(np.float32)
        span = (space.high - space.low).astype(np.float32)
        self._span = np.where(span > 0, span, 1.0).astype(np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        scaled = (observation.astype(np.float32) - self._low) / self._span
        return np.clip(scaled, 0.0, 1.0)


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Traduz overrides opcionais da CLI para argumentos do FloodGuardEnv."""
    env_kwargs: dict[str, Any] = {}
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.flood_advance_prob is not None:
        env_kwargs["flood_advance_prob"] = args.flood_advance_prob
    if args.flood_deepen_prob is not None:
        env_kwargs["flood_deepen_prob"] = args.flood_deepen_prob
    return env_kwargs


def _describe_env(env: FloodGuardEnv) -> dict[str, Any]:
    """Registra os parametros do MDP junto com os resultados do treino."""
    return {
        "grid_size": env.grid_size,
        "max_steps": env.max_steps,
        "max_battery": env.max_battery,
        "initial_barrier_kits": env.initial_barrier_kits,
        "flood_advance_prob": env.flood_advance_prob,
        "flood_deepen_prob": env.flood_deepen_prob,
        "move_cost_dry": env.move_cost_dry,
        "move_cost_shallow": env.move_cost_shallow,
        "barrier_install_cost": env.barrier_install_cost,
        "charge_rate": env.charge_rate,
        "rescue_reward": env.rescue_reward,
        "barrier_effective_reward": env.barrier_effective_reward,
        "step_penalty": env.step_penalty,
        "animal_lost_penalty": env.animal_lost_penalty,
        "battery_depleted_penalty": env.battery_depleted_penalty,
    }


def make_training_env(
    *,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
    monitor_file: Path | None = None,
    reward_shaping: bool = False,
) -> gym.Env:
    """Cria o ambiente com os wrappers esperados pelo stable-baselines3."""
    env = FloodGuardEnv(**(env_kwargs or {}))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # O resgate completo e raro sob exploracao aleatoria. Durante o treino,
    # este wrapper informa se uma acao aproximou o robo do proximo objetivo.
    # A avaliacao usa reward_shaping=False e mede apenas a recompensa do MDP.
    if reward_shaping:
        env = RescueProgressReward(env)

    monitor_path = None
    if monitor_file is not None:
        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        monitor_path = str(monitor_file)

    # Monitor vem antes do FlattenObservation para registrar recompensas,
    # tamanho dos episodios e metricas extras sem alterar a dinamica do MDP.
    env = Monitor(env, filename=monitor_path, info_keywords=MONITOR_INFO_KEYWORDS)

    # FlattenObservation converte o Dict do ambiente em um vetor Box; assim DQN,
    # PPO e A2C podem compartilhar a MlpPolicy default do SB3.
    env = FlattenObservation(env)

    # O vetor achatado mistura mapas 0/1 com bateria e passos que chegam a 100.
    # Colocar todas as entradas em [0, 1] (float32) evita que os campos de maior
    # escala dominem a rede, mantendo intactos o MDP e os hiperparametros default.
    return NormalizeObservation(env)


def build_model(
    *,
    algo: str,
    env: gym.Env,
    seed: int,
    tensorboard_log: Path,
) -> BaseAlgorithm:
    """Instancia o algoritmo escolhido mantendo os hiperparametros default."""
    if algo not in ALGORITHMS:
        raise ValueError(f"Algoritmo invalido: {algo!r}. Use um de {sorted(ALGORITHMS)}.")

    tensorboard_log.mkdir(parents=True, exist_ok=True)
    algorithm_cls = ALGORITHMS[algo]
    return algorithm_cls(
        DEFAULT_POLICY,
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(tensorboard_log),
    )


def _episode_from_info(
    *,
    episode: int,
    seed: int,
    return_total: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> EpisodeMetrics:
    """Converte o info terminal no mesmo formato usado pelo baseline."""
    animal_status = int(info["animal_status"])
    termination_reason = str(info["termination_reason"])
    return EpisodeMetrics(
        episode=episode,
        seed=seed,
        return_total=float(return_total),
        steps=int(info["steps"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=termination_reason,
        animal_status=ANIMAL_STATUS_LABELS[animal_status],
        is_success=bool(info["is_success"]),
        animal_lost=animal_status == FloodGuardEnv.LOST,
        battery_depleted=termination_reason == "battery_depleted",
        final_battery=int(info["battery"]),
        battery_spent=int(info["battery_spent"]),
        barriers_installed=int(info["barriers_installed"]),
        effective_barriers_installed=int(info["effective_barriers_installed"]),
        pickup_step=info["pickup_step"],
        delivery_step=info["delivery_step"],
    )


# Passos consecutivos sem nenhuma mudanca no robo que caracterizam um travamento.
STALL_WINDOW = 10


def _progress_signature(env: FloodGuardEnv) -> tuple[Any, ...]:
    """Estado do robo/animal que muda quando a acao teve algum efeito."""
    return (
        tuple(int(value) for value in env.robot_pos),
        int(env.battery),
        int(env.kits_remaining),
        int(env.animal_status),
    )


def _as_discrete_action(action: Any) -> int:
    """Normaliza a saida do SB3 para a acao discreta esperada pelo ambiente."""
    return int(np.asarray(action).item())


def evaluate_model(
    *,
    model: BaseAlgorithm,
    episodes: int,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    """Avalia a politica treinada em episodios novos.

    O modo padrao e deterministico (acao de maior valor/probabilidade), que e
    a politica que o algoritmo aprendeu e a unica comparavel entre os tres:
    no DQN, `deterministic=False` significa epsilon-greedy com o epsilon final
    do treino, nao uma politica estocastica aprendida.

    Alem das metricas do baseline, o resumo traz um diagnostico de travamento:
    `stall_step_rate` (fracao de passos sem nenhuma mudanca no robo) e
    `stalled_episode_rate` (episodios com >= STALL_WINDOW passos seguidos assim).
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    env = make_training_env(seed=seed, env_kwargs=env_kwargs)
    unwrapped_env = env.unwrapped
    episode_metrics: list[EpisodeMetrics] = []
    stall_steps = 0
    total_steps = 0
    stalled_episodes = 0

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, _ = env.reset(seed=episode_seed)
            total_return = 0.0
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            run_length = 0
            longest_run = 0

            while not (terminated or truncated):
                before = _progress_signature(unwrapped_env)
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(
                    _as_discrete_action(action)
                )
                total_return += float(reward)

                total_steps += 1
                if _progress_signature(unwrapped_env) == before:
                    stall_steps += 1
                    run_length += 1
                    longest_run = max(longest_run, run_length)
                else:
                    run_length = 0
            stalled_episodes += int(longest_run >= STALL_WINDOW)

            episode_metrics.append(
                _episode_from_info(
                    episode=episode,
                    seed=episode_seed,
                    return_total=total_return,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )

        summary = summarize_metrics(
            episode_metrics,
            seed=seed,
            env_params=_describe_env(unwrapped_env),
        )
        summary["deterministic"] = bool(deterministic)
        summary["stall_step_rate"] = stall_steps / max(total_steps, 1)
        summary["stalled_episode_rate"] = stalled_episodes / episodes
        return episode_metrics, summary
    finally:
        env.close()


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _project_relative_path(path: Path) -> str:
    """Representa um artefato sem expor o caminho absoluto da maquina."""
    return os.path.relpath(path.resolve(), start=PROJECT_ROOT)


def train_default_model(args: argparse.Namespace) -> dict[str, Any]:
    """Executa treino, salva modelo e grava um resumo de avaliacao."""
    algo = args.algo.lower()
    seed = set_global_seed(args.seed)
    env_kwargs = _env_kwargs_from_args(args)

    run_name = f"{algo}_seed_{seed}"
    algo_log_dir = args.log_dir / algo
    tensorboard_dir = algo_log_dir / "tensorboard"
    monitor_file = algo_log_dir / f"{run_name}.monitor.csv"
    model_path = args.model_dir / algo / f"floodguard_{run_name}.zip"
    evaluation_path = algo_log_dir / f"{run_name}_evaluation.json"

    train_env = make_training_env(
        seed=seed,
        env_kwargs=env_kwargs,
        monitor_file=monitor_file,
        reward_shaping=True,
    )

    try:
        model = build_model(
            algo=algo,
            env=train_env,
            seed=seed,
            tensorboard_log=tensorboard_dir,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=run_name,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        eval_episodes, evaluation = evaluate_model(
            model=model,
            episodes=args.eval_episodes,
            seed=seed + args.eval_seed_offset,
            env_kwargs=env_kwargs,
            deterministic=not args.stochastic_eval,
        )
        _baseline_episodes, baseline = evaluate_random_policy(
            episodes=args.eval_episodes,
            seed=seed + args.eval_seed_offset,
            env_kwargs=env_kwargs,
        )

        # A Etapa 4 e validada somente se a politica supera o agente aleatorio
        # nas duas medidas centrais: retorno medio e taxa de resgate.
        validation = {
            "return_above_random": evaluation["return_mean"] > baseline["return_mean"],
            "success_above_random": evaluation["success_rate"] > baseline["success_rate"],
        }
        validation["stage4_validated"] = all(validation.values())

        payload = {
            "algo": algo,
            "policy": DEFAULT_POLICY,
            "seed": seed,
            "total_timesteps": args.total_timesteps,
            "env_kwargs": env_kwargs,
            "paths": {
                "model": _project_relative_path(model_path),
                "monitor": _project_relative_path(monitor_file),
                "tensorboard": _project_relative_path(tensorboard_dir),
            },
            "evaluation": evaluation,
            "eval_mode": "stochastic" if args.stochastic_eval else "deterministic",
            "random_baseline": baseline,
            "validation": validation,
            "episodes": [asdict(metrics) for metrics in eval_episodes],
        }
        _write_json(payload, evaluation_path)

        payload["paths"]["evaluation"] = _project_relative_path(evaluation_path)
        return payload
    finally:
        train_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a default SB3 agent on FloodGuardEnv (Stage 4)."
    )
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Training budget passed directly to SB3 learn().",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=10_000,
        help="Offset used to evaluate on seeds different from training.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Evaluate with a sampled/epsilon-greedy policy instead of the default deterministic one.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--flood-advance-prob", type=float, default=None)
    parser.add_argument("--flood-deepen-prob", type=float, default=None)
    return parser


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    # Redes MLP minusculas: varias threads de torch so causam contencao.
    torch.set_num_threads(1)
    args = build_parser().parse_args()
    result = train_default_model(args)
    evaluation = result["evaluation"]

    print("SB3 training complete")
    print(f"Algorithm: {result['algo'].upper()}")
    print(f"Total timesteps: {result['total_timesteps']}")
    print(f"Return mean: {_number(evaluation['return_mean'])} +/- {_number(evaluation['return_std'])}")
    print(f"Success rate: {_percent(evaluation['success_rate'])}")
    print(f"Animal lost rate: {_percent(evaluation['animal_lost_rate'])}")
    print(f"Random baseline return: {_number(result['random_baseline']['return_mean'])}")
    print(f"Stage 4 validated: {result['validation']['stage4_validated']}")
    print(f"Model: {result['paths']['model']}")
    print(f"TensorBoard logs: {result['paths']['tensorboard']}")
    print(f"Evaluation: {result['paths']['evaluation']}")


if __name__ == "__main__":
    main()
