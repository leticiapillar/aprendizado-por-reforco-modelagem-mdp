"""Final multi-seed training and comparison for FloodGuard (Stage 6).

Para cada algoritmo, treina a configuracao otimizada (Etapa 5) e a configuracao
default do SB3 com as mesmas seeds e o mesmo orcamento, e avalia todos os
modelos nos mesmos episodios de teste, comparando com o agente aleatorio.

A avaliacao principal e **deterministica**; a estocastica e reportada como
avaliacao secundaria. Opcionalmente (`--ablations`) treina o PPO otimizado sem
barreiras e sem reward shaping, para investigar o comportamento de protecao.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "floodguard_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

try:
    from experiments.random_baseline import evaluate_random_policy
    from experiments.train import (
        ALGORITHMS,
        DEFAULT_POLICY,
        PROJECT_ROOT,
        STALL_WINDOW,
        _env_kwargs_from_args,
        _number,
        _percent,
        _project_relative_path,
        evaluate_model,
        make_training_env,
    )
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    from random_baseline import evaluate_random_policy  # type: ignore[no-redef]
    from train import (  # type: ignore[no-redef]
        ALGORITHMS,
        DEFAULT_POLICY,
        PROJECT_ROOT,
        STALL_WINDOW,
        _env_kwargs_from_args,
        _number,
        _percent,
        _project_relative_path,
        evaluate_model,
        make_training_env,
    )
from floodguard.envs.flood_guard_env import FloodGuardEnv
from floodguard.utils.seeding import set_global_seed


DEFAULT_SEEDS = (42, 123, 2024, 7, 314)
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "optuna"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "final"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "final"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "relatorio_etapa_6.md"

# Seeds dos episodios de teste (eval_seed .. eval_seed + N) e da avaliacao
# periodica durante o treino (disjuntas entre si).
PERIODIC_EVAL_SEED = 90_000

COMPARISON_METRICS = (
    "return_mean",
    "success_rate",
    "animal_lost_rate",
    "truncated_rate",
    "battery_depleted_rate",
    "battery_spent_mean",
    "steps_to_success_mean",
    "effective_barriers_installed_mean",
    "barriers_installed_mean",
    "stall_step_rate",
    "stalled_episode_rate",
)

VARIANT_LABELS = {
    "tuned": "Otimizado (Optuna)",
    "default": "Default SB3",
    "ppo_no_barriers": "PPO otimizado sem barreiras",
    "ppo_no_shaping": "PPO otimizado sem reward shaping",
}
COLORS = {"a2c": "#2a9d8f", "ppo": "#457b9d", "dqn": "#e76f51", "random": "#6c757d"}


def load_best_params(algo: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / f"{algo}_best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Configuracao da Etapa 5 nao encontrada: {_project_relative_path(path)}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("algo") != algo or not payload.get("best_params"):
        raise ValueError(f"Configuracao invalida para {algo.upper()}: {path}")
    return dict(payload["best_params"])


def build_final_model(
    *,
    algo: str,
    env: Any,
    seed: int,
    tensorboard_log: Path,
    hyperparams: dict[str, Any],
    verbose: int = 0,
) -> Any:
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    return ALGORITHMS[algo](
        DEFAULT_POLICY,
        env,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(tensorboard_log),
        **hyperparams,
    )


class PeriodicEvalCallback(BaseCallback):
    """Avalia a politica (deterministica, recompensa original) durante o treino."""

    def __init__(
        self,
        *,
        eval_freq: int,
        episodes: int,
        env_kwargs: dict[str, Any],
        rows: list[dict[str, Any]],
    ):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.env_kwargs = env_kwargs
        self.rows = rows

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            _episodes, summary = evaluate_model(
                model=self.model,
                episodes=self.episodes,
                seed=PERIODIC_EVAL_SEED,
                env_kwargs=self.env_kwargs,
                deterministic=True,
            )
            self.rows.append(
                {
                    "step": self.num_timesteps,
                    "return_mean": summary["return_mean"],
                    "success_rate": summary["success_rate"],
                }
            )
        return True


def _write_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Nao ha dados para gravar no CSV.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"seeds": len(summaries)}
    for metric in COMPARISON_METRICS:
        values = [float(item[metric]) for item in summaries if item.get(metric) is not None]
        aggregate[f"{metric}_mean"] = mean(values) if values else None
        aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return aggregate


def _learning_curve(monitor: Path, total_timesteps: int, points: int = 100) -> list[dict[str, Any]]:
    """Retorno de treino suavizado, reamostrado em `points` passos igualmente espacados."""
    with monitor.open(encoding="utf-8") as file:
        next(file)
        rows = list(csv.DictReader(file))
    if not rows:
        return []
    steps = np.cumsum([int(row["l"]) for row in rows]).astype(float)
    returns = np.asarray([float(row["r"]) for row in rows])
    window = min(50, len(returns))
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    steps = steps[window - 1 :]
    grid = np.linspace(steps[0], min(steps[-1], total_timesteps), points)
    values = np.interp(grid, steps, smooth)
    return [{"step": int(step), "return_smooth": float(value)} for step, value in zip(grid, values)]


def make_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Uma entrada por (variante, algoritmo, seed)."""
    base_env_kwargs = _env_kwargs_from_args(args)
    tuned = {algo: load_best_params(algo, args.config_dir) for algo in args.algos}
    specs: list[dict[str, Any]] = []
    for algo in args.algos:
        specs.append(
            {"variant": "tuned", "algo": algo, "hyperparams": tuned[algo],
             "env_kwargs": base_env_kwargs, "reward_shaping": True}
        )
        if args.with_defaults:
            specs.append(
                {"variant": "default", "algo": algo, "hyperparams": {},
                 "env_kwargs": base_env_kwargs, "reward_shaping": True}
            )
    if args.ablations and "ppo" in args.algos:
        specs.append(
            {"variant": "ppo_no_barriers", "algo": "ppo", "hyperparams": tuned["ppo"],
             "env_kwargs": {**base_env_kwargs, "initial_barrier_kits": 0},
             "reward_shaping": True}
        )
        specs.append(
            {"variant": "ppo_no_shaping", "algo": "ppo", "hyperparams": tuned["ppo"],
             "env_kwargs": base_env_kwargs, "reward_shaping": False}
        )
    jobs = []
    for spec in specs:
        for seed in args.seeds:
            jobs.append(
                {
                    **spec,
                    "seed": seed,
                    "total_timesteps": args.total_timesteps,
                    "eval_episodes": args.eval_episodes,
                    "eval_seed": args.eval_seed,
                    "curve_eval_freq": args.curve_eval_freq,
                    "curve_eval_episodes": args.curve_eval_episodes,
                    "reuse_models": args.reuse_models,
                    "log_dir": args.log_dir,
                    "model_dir": args.model_dir,
                    "output_dir": args.output_dir,
                    "verbose": args.verbose,
                    "log_interval": args.log_interval,
                    "progress_bar": args.progress_bar,
                }
            )
    return jobs


def run_single(job: dict[str, Any]) -> dict[str, Any]:
    """Treina (ou recarrega) um modelo e o avalia nos modos deterministico e estocastico."""
    torch.set_num_threads(1)
    algo, variant, seed = job["algo"], job["variant"], job["seed"]
    set_global_seed(seed)
    run_name = f"{variant}_{algo}_seed_{seed}"
    monitor = job["log_dir"] / variant / algo / f"{run_name}.monitor.csv"
    tensorboard = job["log_dir"] / variant / algo / "tensorboard"
    model_path = job["model_dir"] / variant / algo / f"floodguard_{run_name}.zip"
    eval_curve: list[dict[str, Any]] = []

    if job["reuse_models"] and model_path.exists():
        model = ALGORITHMS[algo].load(model_path)
    else:
        train_env = make_training_env(
            seed=seed,
            env_kwargs=job["env_kwargs"],
            monitor_file=monitor,
            reward_shaping=job["reward_shaping"],
        )
        try:
            model = build_final_model(
                algo=algo,
                env=train_env,
                seed=seed,
                tensorboard_log=tensorboard,
                hyperparams=job["hyperparams"],
                verbose=job["verbose"],
            )
            callback = PeriodicEvalCallback(
                eval_freq=job["curve_eval_freq"],
                episodes=job["curve_eval_episodes"],
                env_kwargs=job["env_kwargs"],
                rows=eval_curve,
            )
            model.learn(
                total_timesteps=job["total_timesteps"],
                tb_log_name=run_name,
                log_interval=job["log_interval"],
                progress_bar=job["progress_bar"],
                callback=callback,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
        finally:
            train_env.close()

    result: dict[str, Any] = {
        "algo": algo,
        "variant": variant,
        "seed": seed,
        "model": _project_relative_path(model_path),
        "learning_curve": (
            _learning_curve(monitor, job["total_timesteps"]) if monitor.exists() else []
        ),
        "eval_curve": eval_curve,
        "evaluations": {},
        "episodes": [],
    }
    for mode, deterministic in (("deterministic", True), ("stochastic", False)):
        episodes, evaluation = evaluate_model(
            model=model,
            episodes=job["eval_episodes"],
            seed=job["eval_seed"],
            env_kwargs=job["env_kwargs"],
            deterministic=deterministic,
        )
        evaluation_path = job["output_dir"] / variant / algo / f"{run_name}_{mode}.json"
        _write_json(evaluation, evaluation_path)
        result["evaluations"][mode] = evaluation
        result["episodes"].extend(
            {
                "variant": variant,
                "algo": algo,
                "train_seed": seed,
                "eval_mode": mode,
                **asdict(episode),
            }
            for episode in episodes
        )
    return result


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------


def _finish_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _band(ax, runs: list[list[dict[str, Any]]], x: str, y: str, color: str, label: str) -> None:
    """Media +/- desvio entre seeds sobre a grade comum de passos."""
    runs = [run for run in runs if run]
    if not runs:
        return
    length = min(len(run) for run in runs)
    steps = np.asarray([row[x] for row in runs[0][:length]], dtype=float)
    values = np.vstack([[row[y] for row in run[:length]] for run in runs])
    center, spread = values.mean(axis=0), values.std(axis=0)
    ax.plot(steps, center, color=color, label=label, linewidth=2)
    ax.fill_between(steps, center - spread, center + spread, color=color, alpha=0.18)


def plot_learning_curves(results: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in sorted({r["algo"] for r in results}):
        runs = [r["learning_curve"] for r in results if r["algo"] == algo and r["variant"] == "tuned"]
        _band(ax, runs, "step", "return_smooth", COLORS[algo], algo.upper())
    ax.set(title="Curvas de aprendizado (recompensa de treino, com reward shaping)",
           xlabel="Passos de treinamento", ylabel="Retorno medio movel")
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish_figure(path)


def plot_eval_curves(results: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for algo in sorted({r["algo"] for r in results}):
        runs = [r["eval_curve"] for r in results if r["algo"] == algo and r["variant"] == "tuned"]
        _band(axes[0], runs, "step", "return_mean", COLORS[algo], algo.upper())
        _band(axes[1], runs, "step", "success_rate", COLORS[algo], algo.upper())
    axes[0].set(title="Retorno de avaliacao (recompensa original, deterministica)",
                xlabel="Passos de treinamento", ylabel="Retorno medio")
    axes[1].set(title="Taxa de sucesso na avaliacao", xlabel="Passos de treinamento",
                ylabel="Taxa de sucesso", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    return _finish_figure(path)


def plot_return_boxplot(episodes: list[dict[str, Any]], random_returns: list[float], path: Path) -> Path:
    algos = sorted({r["algo"] for r in episodes if r["variant"] == "tuned"})
    data = [random_returns] + [
        [float(r["return_total"]) for r in episodes
         if r["algo"] == a and r["variant"] == "tuned" and r["eval_mode"] == "deterministic"]
        for a in algos
    ]
    labels = ["ALEATORIO"] + [a.upper() for a in algos]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.title("Retorno por episodio de teste (politica deterministica)")
    plt.ylabel("Retorno por episodio")
    plt.grid(axis="y", alpha=0.25)
    return _finish_figure(path)


def plot_final_metrics(aggregates: dict[str, Any], path: Path) -> Path:
    algos = sorted(a for a in aggregates["tuned"])
    labels = ["Sucesso", "Perda do animal", "Truncamento"]
    keys = ["success_rate", "animal_lost_rate", "truncated_rate"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(algos), 1)
    plt.figure(figsize=(9, 4.8))
    for index, algo in enumerate(algos):
        agg = aggregates["tuned"][algo]["deterministic"]
        values = [100.0 * float(agg[f"{key}_mean"] or 0.0) for key in keys]
        errors = [100.0 * float(agg[f"{key}_std"] or 0.0) for key in keys]
        plt.bar(x + (index - (len(algos) - 1) / 2) * width, values, width, yerr=errors,
                capsize=3, label=algo.upper(), color=COLORS[algo])
    plt.xticks(x, labels)
    plt.ylabel("Taxa (%)")
    plt.title("Metricas finais (deterministica, media +/- desvio entre seeds)")
    plt.ylim(0, 108)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    return _finish_figure(path)


def plot_default_vs_tuned(aggregates: dict[str, Any], path: Path) -> Path:
    algos = sorted(aggregates["tuned"])
    if "default" not in aggregates:
        return path
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in (
        (axes[0], "return_mean", "Retorno medio de teste"),
        (axes[1], "success_rate", "Taxa de sucesso"),
    ):
        x = np.arange(len(algos))
        for offset, variant, hatch in ((-0.2, "default", "//"), (0.2, "tuned", None)):
            vals = [aggregates[variant][a]["deterministic"][f"{key}_mean"] for a in algos]
            errs = [aggregates[variant][a]["deterministic"][f"{key}_std"] for a in algos]
            ax.bar(x + offset, vals, 0.4, yerr=errs, capsize=3, hatch=hatch,
                   color=[COLORS[a] for a in algos], edgecolor="white",
                   label=VARIANT_LABELS[variant])
        ax.set_xticks(x, [a.upper() for a in algos])
        ax.set_title(title)
        if key == "success_rate":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Default (hachurado) x otimizado, avaliacao deterministica")
    return _finish_figure(path)


def save_episode_gifs(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    """GIF de um episodio de teste deterministico por algoritmo (modelo otimizado, 1a seed)."""
    paths: dict[str, str] = {}
    env_kwargs = _env_kwargs_from_args(args)
    for algo in args.algos:
        chosen = next((r for r in results if r["algo"] == algo and r["variant"] == "tuned"), None)
        if chosen is None:
            continue
        model = ALGORITHMS[algo].load(PROJECT_ROOT / chosen["model"])
        env = make_training_env(seed=args.eval_seed, env_kwargs=env_kwargs)
        base = env.unwrapped
        base.render_mode = "rgb_array"
        try:
            obs, _ = env.reset(seed=args.eval_seed)
            frames = [base.render()]
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, _info = env.step(int(np.asarray(action).item()))
                frames.append(base.render())
        finally:
            env.close()
        path = args.figure_dir / f"stage6_episode_{algo}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=250, loop=0)
        paths[algo] = _project_relative_path(path)
    return paths


# ----------------------------------------------------------------------
# Relatorio
# ----------------------------------------------------------------------


def _report_relative_path(project_path: str, report_path: Path) -> str:
    absolute = (PROJECT_ROOT / project_path).resolve()
    return os.path.relpath(absolute, start=report_path.parent.resolve())


def _metric_cells(agg: dict[str, Any]) -> list[str]:
    def pm(key: str, percent: bool = False) -> str:
        mean_, std_ = agg.get(f"{key}_mean"), agg.get(f"{key}_std")
        if mean_ is None:
            return "n/a"
        fmt = _percent if percent else _number
        return f"{fmt(mean_)} +/- {fmt(std_ or 0.0)}"

    return [
        pm("return_mean"),
        pm("success_rate", True),
        pm("animal_lost_rate", True),
        pm("truncated_rate", True),
        _number(agg.get("battery_spent_mean_mean")),
        _number(agg.get("steps_to_success_mean_mean")),
    ]


def _random_cells(summary: dict[str, Any]) -> list[str]:
    return [
        f"{_number(summary['return_mean'])} +/- {_number(summary['return_std'])}",
        _percent(summary["success_rate"]),
        _percent(summary["animal_lost_rate"]),
        _percent(summary["truncated_rate"]),
        _number(summary["battery_spent_mean"]),
        _number(summary.get("steps_to_success_mean")),
    ]


TABLE_HEADER = [
    "| Politica | Retorno | Sucesso | Perda do animal | Truncamento | Bateria usada | Passos ate o sucesso |",
    "|---|---:|---:|---:|---:|---:|---:|",
]


def _results_table(summary: dict[str, Any], mode: str) -> list[str]:
    aggregates = summary["aggregates"]
    lines = [*TABLE_HEADER, "| ALEATORIO | " + " | ".join(_random_cells(summary["random_baseline"])) + " |"]
    for variant in ("default", "tuned"):
        for algo in sorted(aggregates.get(variant, {})):
            cells = _metric_cells(aggregates[variant][algo][mode])
            lines.append(f"| {algo.upper()} ({VARIANT_LABELS[variant]}) | " + " | ".join(cells) + " |")
    return lines


def _insights(summary: dict[str, Any]) -> list[str]:
    agg = summary["aggregates"]
    tuned = agg["tuned"]
    det = {a: tuned[a]["deterministic"] for a in tuned}
    ranking = sorted(det, key=lambda a: det[a]["return_mean_mean"], reverse=True)
    random_return = summary["random_baseline"]["return_mean"]
    lines = [
        f"- **Ranking por retorno (deterministica, otimizado):** "
        + " > ".join(f"{a.upper()} ({_number(det[a]['return_mean_mean'])})" for a in ranking)
        + f"; agente aleatorio: {_number(random_return)}."
    ]
    for algo in ranking:
        beats = det[algo]["return_mean_mean"] > random_return
        lines.append(
            f"- **{algo.upper()}:** sucesso {_percent(det[algo]['success_rate_mean'])} "
            f"(+/- {_percent(det[algo]['success_rate_std'])}), "
            f"{'acima' if beats else 'abaixo'} do baseline aleatorio em retorno."
        )
    if "default" in agg:
        for algo in ranking:
            d = agg["default"][algo]["deterministic"]["return_mean_mean"]
            t = det[algo]["return_mean_mean"]
            verdict = "melhorou" if t > d else "nao melhorou"
            lines.append(
                f"- **{algo.upper()}: a otimizacao {verdict}** frente ao default do SB3 "
                f"(retorno {_number(d)} -> {_number(t)})."
            )
    for algo in ranking:
        stall = det[algo]["stalled_episode_rate_mean"]
        stoch = tuned[algo]["stochastic"]["success_rate_mean"]
        lines.append(
            f"- **{algo.upper()}:** {_percent(stall)} dos episodios deterministicos travaram "
            f"(>= {summary['stall_window']} passos sem efeito); sucesso estocastico "
            f"{_percent(stoch)} contra {_percent(det[algo]['success_rate_mean'])} deterministico."
        )
    ref = summary["reference_returns"]
    lines.append(
        f"- **Referencia do MDP:** resgate direto ({ref['path_steps']} passos) rende "
        f"{_number(ref['direct_rescue'])}; instalar {ref['barrier_kits']} barreiras efetivas no "
        f"caminho rende ate {_number(ref['with_effective_barriers'])} (+recompensa de barreira, "
        "-1 passo cada). Retornos acima do resgate direto indicam uso de barreiras."
    )
    barriers = {a: det[a]["effective_barriers_installed_mean_mean"] for a in ranking}
    lines.append(
        "- Barreiras efetivas por episodio (deterministica): "
        + ", ".join(f"{a.upper()} {_number(v)}" for a, v in barriers.items())
        + f"; aleatorio {_number(summary['random_baseline'].get('effective_barriers_installed_mean'))}."
    )
    return lines


def write_report(summary: dict[str, Any], path: Path) -> Path:
    aggregates = summary["aggregates"]
    lines = [
        "# FloodGuard - Relatorio da Etapa 6",
        "",
        f"Gerado em `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` por "
        "`experiments/final_experiments.py`.",
        "",
        "## Resumo",
        "",
        "Cada algoritmo foi treinado com a melhor configuracao da Etapa 5 (\"otimizado\") "
        "e com os hiperparametros default do SB3, usando as mesmas seeds, o mesmo "
        "orcamento e os mesmos episodios de teste, e comparado ao agente aleatorio.",
        "",
        "## Configuracao",
        "",
        f"- Sementes de treino: `{', '.join(map(str, summary['seeds']))}`.",
        f"- Passos de treino por modelo: `{summary['total_timesteps']}`.",
        f"- Episodios de teste por modelo: `{summary['eval_episodes']}` "
        f"(seeds de teste a partir de `{summary['eval_seed']}`; a avaliacao periodica "
        f"durante o treino usa seeds disjuntas a partir de `{PERIODIC_EVAL_SEED}`).",
        "- **Avaliacao principal: deterministica** (acao de maior valor/probabilidade). "
        "E a politica que o algoritmo aprendeu e a unica comparavel entre os tres: no DQN, "
        "o modo estocastico do SB3 e epsilon-greedy com o epsilon final do treino.",
        "- A avaliacao estocastica e reportada como secundaria (robustez).",
        "- A avaliacao usa a recompensa original do MDP, sem o reward shaping do treino.",
        "",
        "## Resultados (avaliacao deterministica)",
        "",
        "Media entre seeds; o termo apos `+/-` e o desvio padrao entre os modelos treinados "
        "(no aleatorio, o desvio entre episodios).",
        "",
        *_results_table(summary, "deterministic"),
        "",
        "## Leitura dos resultados",
        "",
        *_insights(summary),
        "",
        "## Avaliacao estocastica (secundaria)",
        "",
        "PPO/A2C amostram da distribuicao de acoes; o DQN usa epsilon-greedy com o epsilon "
        "final do treino. Os modos nao sao equivalentes entre algoritmos; a tabela serve "
        "para medir a sensibilidade a ruido de politica.",
        "",
        *_results_table(summary, "stochastic"),
        "",
        "## Diagnostico de travamento (deterministica)",
        "",
        f"Um passo e \"sem efeito\" quando robo, bateria, kits e status do animal nao mudam. "
        f"Um episodio \"trava\" com ao menos {summary['stall_window']} passos seguidos assim.",
        "",
        "| Politica | Passos sem efeito | Episodios travados |",
        "|---|---:|---:|",
    ]
    for variant in ("default", "tuned"):
        for algo in sorted(aggregates.get(variant, {})):
            agg = aggregates[variant][algo]["deterministic"]
            lines.append(
                f"| {algo.upper()} ({VARIANT_LABELS[variant]}) | "
                f"{_percent(agg['stall_step_rate_mean'])} | {_percent(agg['stalled_episode_rate_mean'])} |"
            )
    ablations = [v for v in aggregates if v.startswith("ppo_")]
    if ablations:
        lines.extend(
            [
                "",
                "## Ablacoes do PPO (protecao vs. resgate)",
                "",
                "Investigam por que as barreiras quase nao sao usadas: sem kits de barreira "
                "e sem o reward shaping de progresso (deterministica).",
                "",
                "| Variante | Retorno | Sucesso | Perda do animal | Truncamento | Barreiras efetivas |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        rows = [("PPO otimizado (referencia)", aggregates["tuned"]["ppo"]["deterministic"])] + [
            (VARIANT_LABELS[v], aggregates[v]["ppo"]["deterministic"]) for v in ablations
        ]
        for label, agg in rows:
            cells = _metric_cells(agg)
            lines.append(
                f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                f"{_number(agg['effective_barriers_installed_mean_mean'])} |"
            )
    lines.extend(["", "## Graficos", ""])
    captions = {
        "learning_curves": "Curvas de aprendizado (recompensa de treino, inclui shaping)",
        "eval_curves": "Avaliacao periodica durante o treino (recompensa original, deterministica)",
        "return_boxplot": "Retorno nos episodios de teste",
        "final_metrics": "Taxas finais",
        "default_vs_tuned": "Default x otimizado",
    }
    for key, caption in captions.items():
        if key in summary["figures"]:
            lines.extend(
                [f"### {caption}", "", f"![{caption}]({_report_relative_path(summary['figures'][key], path)})", ""]
            )
    if summary.get("episode_gifs"):
        lines.extend(["### Episodio de teste por algoritmo (modelo otimizado, deterministica)", ""])
        for algo, gif in sorted(summary["episode_gifs"].items()):
            lines.extend([f"**{algo.upper()}**", "", f"![Episodio {algo.upper()}]({_report_relative_path(gif, path)})", ""])
    lines.extend(
        [
            "## Arquivos gerados",
            "",
            f"- Modelos: `{summary['paths']['models']}`.",
            f"- Resultados por seed: `{summary['paths']['seed_results']}`.",
            f"- Resultados por episodio: `{summary['paths']['episode_results']}`.",
            f"- Curvas de treino e de avaliacao: `{summary['paths']['learning_curves']}`, `{summary['paths']['eval_curves']}`.",
            f"- Resumo em JSON: `{summary['paths']['summary']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Orquestracao
# ----------------------------------------------------------------------


def load_results(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Reconstroi `results` a partir dos arquivos salvos (modo `--report-only`)."""
    out = args.output_dir

    def read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def key(row: dict[str, str]) -> tuple[str, str, int]:
        return row["variant"], row["algo"], int(row.get("train_seed") or row["seed"])

    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read(out / "final_seed_results.csv"):
        variant, algo, seed = key(row)
        run_name = f"{variant}_{algo}_seed_{seed}"
        entry = results.setdefault(
            (variant, algo, seed),
            {
                "algo": algo, "variant": variant, "seed": seed,
                "model": _project_relative_path(
                    args.model_dir / variant / algo / f"floodguard_{run_name}.zip"),
                "learning_curve": [], "eval_curve": [], "evaluations": {}, "episodes": [],
            },
        )
        mode = row["eval_mode"]
        entry["evaluations"][mode] = json.loads(
            (out / variant / algo / f"{run_name}_{mode}.json").read_text(encoding="utf-8"))
    for row in read(out / "final_episode_results.csv"):
        results[key(row)]["episodes"].append(
            {**row, "return_total": float(row["return_total"])})
    for name, field, cols in (
        ("learning_curves.csv", "learning_curve", ("return_smooth",)),
        ("eval_curves.csv", "eval_curve", ("return_mean", "success_rate")),
    ):
        path = out / name
        if path.exists():
            for row in read(path):
                results[key(row)][field].append(
                    {"step": int(row["step"]), **{c: float(row[c]) for c in cols}})
    return list(results.values())


def run_final_experiments(args: argparse.Namespace) -> dict[str, Any]:
    if args.report_only:
        results = load_results(args)
    else:
        jobs = make_jobs(args)
        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                results = list(pool.map(run_single, jobs))
        else:
            results = [run_single(job) for job in jobs]
    return finalize(args, results)


def reference_returns(env_kwargs: dict[str, Any]) -> dict[str, float]:
    """Retornos de referencia do MDP: resgate direto e resgate com barreiras efetivas."""
    env = FloodGuardEnv(**env_kwargs)
    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    path = manhattan(env.robot_base, env.animal_start) + manhattan(env.animal_start, env.safe_zone)
    direct = env.rescue_reward + env.step_penalty * path
    kits = env.initial_barrier_kits
    with_barriers = direct + kits * (env.barrier_effective_reward + env.step_penalty)
    return {"path_steps": path, "direct_rescue": direct, "with_effective_barriers": with_barriers,
            "barrier_kits": kits}


def finalize(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    env_kwargs = _env_kwargs_from_args(args)
    seed_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        episode_rows.extend(result["episodes"])
        key = {"variant": result["variant"], "algo": result["algo"], "seed": result["seed"]}
        curve_rows.extend({**key, **row} for row in result["learning_curve"])
        eval_rows.extend({**key, **row} for row in result["eval_curve"])
        for mode, evaluation in result["evaluations"].items():
            seed_rows.append(
                {**key, "eval_mode": mode,
                 **{m: evaluation.get(m) for m in COMPARISON_METRICS}}
            )
            grouped.setdefault((result["variant"], result["algo"], mode), []).append(evaluation)

    aggregates: dict[str, Any] = {}
    for (variant, algo, mode), summaries in grouped.items():
        aggregates.setdefault(variant, {}).setdefault(algo, {})[mode] = _aggregate_seed_summaries(summaries)

    random_episodes, random_summary = evaluate_random_policy(
        episodes=args.eval_episodes, seed=args.eval_seed, env_kwargs=env_kwargs
    )

    out = args.output_dir
    seed_csv = _write_csv(seed_rows, out / "final_seed_results.csv")
    episode_csv = _write_csv(episode_rows, out / "final_episode_results.csv")
    curves_csv = _write_csv(curve_rows, out / "learning_curves.csv") if curve_rows else out / "learning_curves.csv"
    eval_csv = _write_csv(eval_rows, out / "eval_curves.csv") if eval_rows else out / "eval_curves.csv"

    figures = {
        "learning_curves": plot_learning_curves(results, args.figure_dir / "stage6_learning_curves.png"),
        "eval_curves": plot_eval_curves(results, args.figure_dir / "stage6_eval_curves.png"),
        "return_boxplot": plot_return_boxplot(
            episode_rows, [ep.return_total for ep in random_episodes],
            args.figure_dir / "stage6_return_boxplot.png"),
        "final_metrics": plot_final_metrics(aggregates, args.figure_dir / "stage6_final_metrics.png"),
    }
    if "default" in aggregates:
        figures["default_vs_tuned"] = plot_default_vs_tuned(
            aggregates, args.figure_dir / "stage6_default_vs_tuned.png")
    episode_gifs = save_episode_gifs(results, args) if args.render_episodes else {}

    summary_path = out / "final_summary.json"
    summary = {
        "aggregates": aggregates,
        "random_baseline": random_summary,
        "configs": {a: load_best_params(a, args.config_dir) for a in args.algos},
        "seeds": list(args.seeds),
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "primary_eval_mode": "deterministic",
        "stall_window": STALL_WINDOW,
        "env_kwargs": env_kwargs,
        "reference_returns": reference_returns(env_kwargs),
        "runs": [
            {k: r[k] for k in ("algo", "variant", "seed", "model")}
            | {"evaluations": r["evaluations"]}
            for r in results
        ],
        "figures": {k: _project_relative_path(v) for k, v in figures.items()},
        "episode_gifs": episode_gifs,
        "paths": {
            "models": _project_relative_path(args.model_dir),
            "seed_results": _project_relative_path(seed_csv),
            "episode_results": _project_relative_path(episode_csv),
            "learning_curves": _project_relative_path(curves_csv),
            "eval_curves": _project_relative_path(eval_csv),
            "summary": _project_relative_path(summary_path),
        },
    }
    _write_json(summary, summary_path)
    report = write_report(summary, args.report_path)
    return {"summary": summary, "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FloodGuard Stage 6 experiments.")
    parser.add_argument("--algos", nargs="+", choices=sorted(ALGORITHMS), default=sorted(ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=50_000)
    parser.add_argument("--curve-eval-freq", type=int, default=25_000,
                        help="Timesteps between periodic evaluations during training (0 disables).")
    parser.add_argument("--curve-eval-episodes", type=int, default=30)
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild figures and report from the saved CSV/JSON files (no training).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel training processes.")
    parser.add_argument("--no-defaults", dest="with_defaults", action="store_false",
                        help="Skip the SB3-default comparison runs.")
    parser.add_argument("--ablations", action="store_true",
                        help="Also train PPO without barriers and without reward shaping.")
    parser.add_argument("--no-render", dest="render_episodes", action="store_false",
                        help="Skip the per-algorithm episode GIFs.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Skip training when the expected model files already exist.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--flood-advance-prob", type=float, default=None)
    parser.add_argument("--flood-deepen-prob", type=float, default=None)
    return parser


def main() -> None:
    # Redes MLP minusculas: varias threads de torch so causam contencao.
    torch.set_num_threads(1)
    args = build_parser().parse_args()
    result = run_final_experiments(args)
    print("Experimentos da Etapa 6 concluidos.")
    print(f"Relatorio: {_project_relative_path(result['report'])}")


if __name__ == "__main__":
    main()
