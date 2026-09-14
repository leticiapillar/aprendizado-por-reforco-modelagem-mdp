"""Random-agent baseline and environment sanity check (Stage 3).

The script runs a uniformly random policy in FloodGuardEnv, aggregates the
metrics that will be reused in the final experiments, and writes three
artifacts under results/baselines/:

  - random_baseline_episodes.csv: one row per episode
  - random_baseline_summary.json: machine-readable aggregate metrics
  - random_baseline_report.md: short human-readable report

Usage:
    python experiments/random_baseline.py --episodes 1000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from floodguard.envs.flood_guard_env import FloodGuardEnv
from floodguard.utils.seeding import DEFAULT_SEED, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"

ANIMAL_STATUS_LABELS = {
    FloodGuardEnv.AWAITING: "awaiting",
    FloodGuardEnv.WITH_ROBOT: "with_robot",
    FloodGuardEnv.SAVED: "saved",
    FloodGuardEnv.LOST: "lost",
}


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    seed: int
    return_total: float
    steps: int
    terminated: bool
    truncated: bool
    termination_reason: str
    animal_status: str
    is_success: bool
    animal_lost: bool
    battery_depleted: bool
    final_battery: int
    battery_spent: int
    barriers_installed: int
    effective_barriers_installed: int
    pickup_step: int | None
    delivery_step: int | None


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {}
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.flood_advance_prob is not None:
        env_kwargs["flood_advance_prob"] = args.flood_advance_prob
    if args.flood_deepen_prob is not None:
        env_kwargs["flood_deepen_prob"] = args.flood_deepen_prob
    return env_kwargs


def _describe_env(env: FloodGuardEnv) -> dict[str, Any]:
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


def run_random_episode(env: FloodGuardEnv, episode: int, seed: int) -> EpisodeMetrics:
    env.reset(seed=seed)
    env.action_space.seed(seed)

    total_return = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        action = int(env.action_space.sample())
        _obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward

    animal_status = int(info["animal_status"])
    termination_reason = str(info["termination_reason"])

    return EpisodeMetrics(
        episode=episode,
        seed=seed,
        return_total=float(total_return),
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


def evaluate_random_policy(
    episodes: int = 1000,
    seed: int = DEFAULT_SEED,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    set_global_seed(seed)
    env = FloodGuardEnv(**(env_kwargs or {}))
    episode_metrics = [
        run_random_episode(env, episode=i, seed=seed + i)
        for i in range(episodes)
    ]
    summary = summarize_metrics(episode_metrics, seed=seed, env_params=_describe_env(env))
    return episode_metrics, summary


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_metrics(
    episode_metrics: list[EpisodeMetrics],
    seed: int,
    env_params: dict[str, Any],
) -> dict[str, Any]:
    n = len(episode_metrics)
    returns = np.array([m.return_total for m in episode_metrics], dtype=float)
    steps = np.array([m.steps for m in episode_metrics], dtype=float)
    battery_spent = np.array([m.battery_spent for m in episode_metrics], dtype=float)
    barriers = np.array([m.barriers_installed for m in episode_metrics], dtype=float)
    effective_barriers = np.array([m.effective_barriers_installed for m in episode_metrics], dtype=float)
    success_delivery_steps = [m.delivery_step for m in episode_metrics if m.is_success and m.delivery_step is not None]
    pickup_steps = [m.pickup_step for m in episode_metrics if m.pickup_step is not None]

    termination_counts = Counter(m.termination_reason for m in episode_metrics)
    status_counts = Counter(m.animal_status for m in episode_metrics)

    return {
        "episodes": n,
        "seed": seed,
        "env_params": env_params,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)) if n > 1 else 0.0,
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "success_rate": sum(m.is_success for m in episode_metrics) / n,
        "animal_lost_rate": sum(m.animal_lost for m in episode_metrics) / n,
        "truncated_rate": sum(m.truncated for m in episode_metrics) / n,
        "battery_depleted_rate": sum(m.battery_depleted for m in episode_metrics) / n,
        "pickup_rate": len(pickup_steps) / n,
        "steps_mean": float(np.mean(steps)),
        "steps_to_success_mean": _mean([float(s) for s in success_delivery_steps]),
        "steps_to_pickup_mean": _mean([float(s) for s in pickup_steps]),
        "battery_spent_mean": float(np.mean(battery_spent)),
        "final_battery_mean": float(np.mean([m.final_battery for m in episode_metrics])),
        "barriers_installed_mean": float(np.mean(barriers)),
        "effective_barriers_installed_mean": float(np.mean(effective_barriers)),
        "termination_counts": dict(termination_counts),
        "animal_status_counts": dict(status_counts),
    }


def write_episode_csv(episode_metrics: list[EpisodeMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(episode_metrics[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in episode_metrics:
            writer.writerow(asdict(metrics))


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(summary: dict[str, Any], output_dir: Path) -> Path:
    env = summary["env_params"]
    report_path = output_dir / "random_baseline_report.md"
    csv_name = "random_baseline_episodes.csv"
    json_name = "random_baseline_summary.json"

    calibration_note = (
        "A calibracao adotada nesta etapa reduziu `flood_advance_prob` para "
        f"`{env['flood_advance_prob']}` e manteve `flood_deepen_prob` em "
        f"`{env['flood_deepen_prob']}`. Com isso, o agente aleatorio tem "
        "sucesso baixo, mas nao nulo, enquanto a perda do animal continua "
        "frequente o suficiente para o ambiente nao ficar trivial."
    )

    lines = [
        "# FloodGuard - Etapa 3: baseline aleatorio",
        "",
        "Relatorio gerado por `python experiments/random_baseline.py`.",
        "",
        "## Parametros do ambiente",
        "",
        "| Parametro | Valor |",
        "|---|---:|",
        f"| Episodios | {summary['episodes']} |",
        f"| Seed inicial | {summary['seed']} |",
        f"| Grid | {env['grid_size']}x{env['grid_size']} |",
        f"| Max steps | {env['max_steps']} |",
        f"| Bateria maxima | {env['max_battery']} |",
        f"| Kits de barreira | {env['initial_barrier_kits']} |",
        f"| Prob. avanco agua rasa (`p`) | {env['flood_advance_prob']} |",
        f"| Prob. aprofundamento (`q`) | {env['flood_deepen_prob']} |",
        "",
        "## Resultados do agente aleatorio",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
        f"| Retorno medio | {_number(summary['return_mean'])} +/- {_number(summary['return_std'])} |",
        f"| Taxa de sucesso | {_percent(summary['success_rate'])} |",
        f"| Taxa de perda do animal | {_percent(summary['animal_lost_rate'])} |",
        f"| Taxa de truncamento por steps | {_percent(summary['truncated_rate'])} |",
        f"| Taxa de bateria zerada fora da base | {_percent(summary['battery_depleted_rate'])} |",
        f"| Taxa de coleta do animal | {_percent(summary['pickup_rate'])} |",
        f"| Steps medios por episodio | {_number(summary['steps_mean'])} |",
        f"| Steps medios ate sucesso | {_number(summary['steps_to_success_mean'])} |",
        f"| Uso medio de bateria | {_number(summary['battery_spent_mean'])} |",
        f"| Barreiras instaladas por episodio | {_number(summary['barriers_installed_mean'])} |",
        f"| Barreiras efetivas por episodio | {_number(summary['effective_barriers_installed_mean'])} |",
        "",
        "## Sanity check",
        "",
        calibration_note,
        "",
        "## Metricas fixadas para as proximas etapas",
        "",
        "- Retorno medio por episodio (`return_mean`).",
        "- Taxa de sucesso: animal entregue na zona segura (`success_rate`).",
        "- Taxa de perda do animal por agua profunda (`animal_lost_rate`).",
        "- Taxa de episodios truncados pelo limite de steps (`truncated_rate`).",
        "- Taxa de bateria zerada fora da base (`battery_depleted_rate`).",
        "- Uso medio de bateria: energia acumulada gasta em movimento/barreiras (`battery_spent_mean`).",
        "- Numero medio de steps ate sucesso, medido apenas nos episodios bem-sucedidos (`steps_to_success_mean`).",
        "- Numero medio de barreiras instaladas e de barreiras efetivas (`barriers_installed_mean`, `effective_barriers_installed_mean`).",
        "",
        "## Dicionario das metricas extras do `info`",
        "",
        "| Metrica | Significado |",
        "|---|---|",
        "| `battery_spent` | Energia acumulada gasta em movimentos e instalacao de barreiras. Difere da bateria final porque o robo pode recarregar na base. |",
        "| `termination_reason` | Motivo do fim do episodio: `success`, `animal_lost`, `battery_depleted`, `max_steps` ou `running` enquanto o episodio ainda nao terminou. |",
        "| `barriers_installed` | Numero de barreiras realmente instaladas no episodio. Acoes invalidas de instalacao nao entram nessa contagem. |",
        "| `effective_barriers_installed` | Numero de barreiras instaladas em celulas com risco imediato: celula rasa ou celula seca adjacente a agua. |",
        "| `pickup_step` | Step em que o robo pegou o animal pela primeira vez. Fica `None` se o animal nunca foi coletado. |",
        "| `delivery_step` | Step em que o robo entregou o animal na zona segura. Fica `None` se nao houve sucesso. |",
        "",
        "## Arquivos gerados",
        "",
        f"- `{csv_name}`: metricas por episodio.",
        f"- `{json_name}`: resumo agregado em formato legivel por codigo.",
        "- `random_baseline_report.md`: este relatorio.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_outputs(
    episode_metrics: list[EpisodeMetrics],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "random_baseline_episodes.csv"
    json_path = output_dir / "random_baseline_summary.json"
    write_episode_csv(episode_metrics, csv_path)
    write_summary_json(summary, json_path)
    report_path = write_report(summary, output_dir)
    return {"csv": csv_path, "json": json_path, "report": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FloodGuard random-agent baseline.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of random episodes to run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Initial seed; episode i uses seed+i.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON/report.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override FloodGuardEnv max_steps.")
    parser.add_argument("--flood-advance-prob", type=float, default=None, help="Override FloodGuardEnv p.")
    parser.add_argument("--flood-deepen-prob", type=float, default=None, help="Override FloodGuardEnv q.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_metrics, summary = evaluate_random_policy(
        episodes=args.episodes,
        seed=args.seed,
        env_kwargs=_env_kwargs_from_args(args),
    )
    paths = save_outputs(episode_metrics, summary, args.output_dir)

    print("Random baseline complete")
    print(f"Episodes: {summary['episodes']}")
    print(f"Return mean: {_number(summary['return_mean'])} +/- {_number(summary['return_std'])}")
    print(f"Success rate: {_percent(summary['success_rate'])}")
    print(f"Animal lost rate: {_percent(summary['animal_lost_rate'])}")
    print(f"Truncated rate: {_percent(summary['truncated_rate'])}")
    print(f"Battery depleted rate: {_percent(summary['battery_depleted_rate'])}")
    print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
