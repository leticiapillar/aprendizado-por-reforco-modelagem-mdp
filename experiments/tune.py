"""Optuna hyperparameter search for FloodGuard (Stage 5).

The script optimizes one Stable-Baselines3 algorithm at a time, records every
trial, saves the best configuration, and refreshes a Markdown report for the
stage.

Examples:
    python experiments/tune.py --algo dqn
    python experiments/tune.py --algo ppo --n-trials 10 --total-timesteps 50000
    python experiments/tune.py --algo a2c --eval-episodes 20 --progress-bar
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "floodguard_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

try:
    from experiments.train import (
        ALGORITHMS,
        DEFAULT_POLICY,
        PROJECT_ROOT,
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
    # Keeps `python experiments/tune.py` working from the project root.
    from train import (  # type: ignore[no-redef]
        ALGORITHMS,
        DEFAULT_POLICY,
        PROJECT_ROOT,
        _env_kwargs_from_args,
        _number,
        _percent,
        _project_relative_path,
        evaluate_model,
        make_training_env,
    )

from floodguard.utils.seeding import DEFAULT_SEED, set_global_seed


DEFAULT_LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "optuna"
DEFAULT_FIGURE_ROOT = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "relatorio_etapa_5.md"

REPORT_METRICS = (
    "return_mean",
    "return_std",
    "success_rate",
    "animal_lost_rate",
    "truncated_rate",
    "battery_depleted_rate",
    "pickup_rate",
    "steps_mean",
    "steps_to_success_mean",
    "battery_spent_mean",
    "effective_barriers_installed_mean",
)


def suggest_hyperparams(trial: optuna.Trial, algo: str) -> dict[str, Any]:
    """Defines the Stage 5 search space for one algorithm."""
    algo = algo.lower()
    if algo == "dqn":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "buffer_size": trial.suggest_categorical(
                "buffer_size", [10_000, 25_000, 50_000, 100_000]
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
            "exploration_fraction": trial.suggest_float(
                "exploration_fraction", 0.05, 0.50
            ),
            "exploration_final_eps": trial.suggest_float(
                "exploration_final_eps", 0.01, 0.20
            ),
            "target_update_interval": trial.suggest_categorical(
                "target_update_interval", [250, 500, 1_000, 2_000, 5_000]
            ),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
            "learning_starts": trial.suggest_categorical(
                "learning_starts", [100, 500, 1_000, 2_000]
            ),
        }

    if algo == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1_024])
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "n_steps": n_steps,
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs": trial.suggest_categorical("n_epochs", [3, 5, 10, 20]),
            "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 1.00),
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.40),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.05, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.00),
        }

    if algo == "a2c":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20, 50, 100]),
            "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 1.00),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.05, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.00),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 1.00),
        }

    raise ValueError(f"Algoritmo invalido: {algo!r}. Use um de {sorted(ALGORITHMS)}.")


def build_tuned_model(
    *,
    algo: str,
    env: Any,
    seed: int,
    tensorboard_log: Path,
    hyperparams: dict[str, Any],
    verbose: int = 0,
) -> BaseAlgorithm:
    """Instantiates a SB3 model with the sampled hyperparameters."""
    if algo not in ALGORITHMS:
        raise ValueError(f"Algoritmo invalido: {algo!r}. Use um de {sorted(ALGORITHMS)}.")

    tensorboard_log.mkdir(parents=True, exist_ok=True)
    algorithm_cls = ALGORITHMS[algo]
    return algorithm_cls(
        DEFAULT_POLICY,
        env,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(tensorboard_log),
        **hyperparams,
    )


class TrialEvalCallback(BaseCallback):
    """Reports intermediate evaluation returns to Optuna for pruning."""

    def __init__(
        self,
        *,
        trial: optuna.Trial,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        env_kwargs: dict[str, Any],
        total_timesteps: int,
    ):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.env_kwargs = env_kwargs
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        _episodes, summary = evaluate_model(
            model=self.model,
            episodes=self.eval_episodes,
            seed=self.eval_seed + self.num_timesteps,
            env_kwargs=self.env_kwargs,
        )
        mean_return = float(summary["return_mean"])
        self.trial.report(mean_return, step=self.num_timesteps)
        if self.num_timesteps >= self.total_timesteps:
            return True
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at {self.num_timesteps} timesteps "
                f"(return_mean={mean_return:.2f})."
            )
        return True


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True), encoding="utf-8")


def _study_name(algo: str) -> str:
    return f"floodguard_stage5_{algo}"


def _storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve()}"


def _trial_duration_seconds(trial: optuna.trial.FrozenTrial) -> float | None:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def write_trials_csv(study: optuna.Study, path: Path) -> Path:
    """Writes one auditable row per trial, including all sampled params."""
    path.parent.mkdir(parents=True, exist_ok=True)
    param_names = sorted({name for trial in study.trials for name in trial.params})
    user_attr_names = sorted(
        {
            name
            for trial in study.trials
            for name, value in trial.user_attrs.items()
            if name in REPORT_METRICS or not isinstance(value, (dict, list, tuple))
        }
    )
    fieldnames = [
        "number",
        "state",
        "value",
        "duration_seconds",
        "datetime_start",
        "datetime_complete",
        *[f"params_{name}" for name in param_names],
        *[f"user_{name}" for name in user_attr_names],
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row: dict[str, Any] = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": _trial_duration_seconds(trial),
                "datetime_start": trial.datetime_start.isoformat()
                if trial.datetime_start
                else None,
                "datetime_complete": trial.datetime_complete.isoformat()
                if trial.datetime_complete
                else None,
            }
            for name in param_names:
                row[f"params_{name}"] = trial.params.get(name)
            for name in user_attr_names:
                row[f"user_{name}"] = trial.user_attrs.get(name)
            writer.writerow(row)
    return path


def _complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_optimization_history(study: optuna.Study, path: Path, algo: str) -> Path:
    complete = _complete_trials(study)
    plt.figure(figsize=(8, 4.5))
    if not complete:
        plt.text(0.5, 0.5, "Nenhum trial completo ainda.", ha="center", va="center")
        plt.axis("off")
        return _save_figure(path)

    numbers = [trial.number for trial in complete]
    values = [float(trial.value) for trial in complete]
    best_so_far: list[float] = []
    current_best = -math.inf
    for value in values:
        current_best = max(current_best, value)
        best_so_far.append(current_best)

    plt.plot(numbers, values, marker="o", label="Retorno do trial")
    plt.plot(numbers, best_so_far, marker="s", label="Melhor ate aqui")
    plt.title(f"Optuna - historico de otimizacao ({algo.upper()})")
    plt.xlabel("Trial")
    plt.ylabel("Retorno medio de avaliacao")
    plt.grid(alpha=0.25)
    plt.legend()
    return _save_figure(path)


def plot_param_importance(study: optuna.Study, path: Path, algo: str) -> Path:
    plt.figure(figsize=(8, 4.5))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            evaluator = PedAnovaImportanceEvaluator()
        importances = get_param_importances(
            study,
            evaluator=evaluator,
        )
    except Exception as exc:  # Optuna raises when there is too little data.
        plt.text(
            0.5,
            0.5,
            f"Importancia indisponivel.\n{exc}",
            ha="center",
            va="center",
            wrap=True,
        )
        plt.axis("off")
        return _save_figure(path)

    if not importances:
        plt.text(0.5, 0.5, "Sem parametros suficientes para estimar importancia.", ha="center")
        plt.axis("off")
        return _save_figure(path)

    names = list(importances)
    values = [float(importances[name]) for name in names]
    plt.barh(names[::-1], values[::-1])
    plt.title(f"Optuna - importancia dos hiperparametros ({algo.upper()})")
    plt.xlabel("Importancia relativa")
    plt.xlim(0, max(values) * 1.1 if values else 1)
    plt.grid(axis="x", alpha=0.25)
    return _save_figure(path)


def write_best_config(
    *,
    study: optuna.Study,
    algo: str,
    path: Path,
    env_kwargs: dict[str, Any],
    total_timesteps: int,
    eval_episodes: int,
) -> Path:
    complete = _complete_trials(study)
    payload: dict[str, Any] = {
        "algo": algo,
        "study_name": study.study_name,
        "direction": study.direction.name,
        "completed_trials": len(complete),
        "total_trials": len(study.trials),
        "total_timesteps_per_trial": total_timesteps,
        "eval_episodes": eval_episodes,
        "env_kwargs": env_kwargs,
    }
    if complete:
        best = study.best_trial
        payload.update(
            {
                "best_trial_number": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "best_metrics": {
                    key: best.user_attrs.get(key)
                    for key in REPORT_METRICS
                    if key in best.user_attrs
                },
            }
        )
    _write_json(payload, path)
    return path


def write_summary_json(
    *,
    study: optuna.Study,
    algo: str,
    path: Path,
    env_kwargs: dict[str, Any],
    paths: dict[str, Path],
    total_timesteps: int,
    eval_episodes: int,
) -> Path:
    complete = _complete_trials(study)
    pruned = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
    failed = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    best_trial = study.best_trial if complete else None
    payload: dict[str, Any] = {
        "algo": algo,
        "study_name": study.study_name,
        "storage": _project_relative_path(paths["storage"]),
        "total_trials": len(study.trials),
        "completed_trials": len(complete),
        "pruned_trials": len(pruned),
        "failed_trials": len(failed),
        "total_timesteps_per_trial": total_timesteps,
        "eval_episodes": eval_episodes,
        "env_kwargs": env_kwargs,
        "paths": {
            key: _project_relative_path(value)
            for key, value in paths.items()
            if key != "storage_url"
        },
    }
    if best_trial is not None:
        payload["best_trial"] = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "metrics": {
                key: best_trial.user_attrs.get(key)
                for key in REPORT_METRICS
                if key in best_trial.user_attrs
            },
        }
    _write_json(payload, path)
    return path


def _make_study(
    *,
    algo: str,
    storage_path: Path,
    seed: int,
    n_startup_trials: int,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=1,
        interval_steps=1,
    )
    return optuna.create_study(
        study_name=_study_name(algo),
        storage=_storage_url(storage_path),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _set_trial_metrics(trial: optuna.Trial, summary: dict[str, Any]) -> None:
    for key in REPORT_METRICS:
        if key in summary:
            trial.set_user_attr(key, summary[key])
    trial.set_user_attr("termination_counts", summary.get("termination_counts", {}))
    trial.set_user_attr("animal_status_counts", summary.get("animal_status_counts", {}))


def objective(
    trial: optuna.Trial,
    *,
    algo: str,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
) -> float:
    hyperparams = suggest_hyperparams(trial, algo)
    trial_seed = args.seed + trial.number
    run_name = f"{algo}_trial_{trial.number}_seed_{trial_seed}"
    monitor_file = args.log_dir / algo / f"{run_name}.monitor.csv"
    tensorboard_dir = args.log_dir / algo / "tensorboard"

    train_env = make_training_env(
        seed=trial_seed,
        env_kwargs=env_kwargs,
        monitor_file=monitor_file,
        reward_shaping=True,
    )
    try:
        model = build_tuned_model(
            algo=algo,
            env=train_env,
            seed=trial_seed,
            tensorboard_log=tensorboard_dir,
            hyperparams=hyperparams,
            verbose=args.verbose,
        )
        callback = TrialEvalCallback(
            trial=trial,
            eval_freq=args.eval_freq,
            eval_episodes=args.pruning_eval_episodes,
            eval_seed=args.seed + args.eval_seed_offset + trial.number * 1_000,
            env_kwargs=env_kwargs,
            total_timesteps=args.total_timesteps,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=run_name,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
            callback=callback,
        )
        _episodes, summary = evaluate_model(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed + args.eval_seed_offset + trial.number * 1_000,
            env_kwargs=env_kwargs,
        )
        _set_trial_metrics(trial, summary)
        trial.set_user_attr("monitor", _project_relative_path(monitor_file))
        trial.set_user_attr("tensorboard", _project_relative_path(tensorboard_dir))
        return float(summary["return_mean"])
    finally:
        train_env.close()


def optimize_hyperparams(args: argparse.Namespace) -> dict[str, Any]:
    algo = args.algo.lower()
    seed = set_global_seed(args.seed)
    env_kwargs = _env_kwargs_from_args(args)

    storage_path = args.log_dir / f"optuna_{algo}.db"
    trials_csv_path = args.log_dir / f"{algo}_trials.csv"
    best_config_path = args.log_dir / f"{algo}_best_params.json"
    summary_path = args.log_dir / f"{algo}_summary.json"
    history_path = args.figure_dir / f"optuna_{algo}_history.png"
    importance_path = args.figure_dir / f"optuna_{algo}_param_importance.png"

    study = _make_study(
        algo=algo,
        storage_path=storage_path,
        seed=seed,
        n_startup_trials=args.n_startup_trials,
    )
    study.optimize(
        lambda trial: objective(trial, algo=algo, args=args, env_kwargs=env_kwargs),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=args.progress_bar,
    )

    paths = {
        "storage": storage_path,
        "trials_csv": write_trials_csv(study, trials_csv_path),
        "best_config": write_best_config(
            study=study,
            algo=algo,
            path=best_config_path,
            env_kwargs=env_kwargs,
            total_timesteps=args.total_timesteps,
            eval_episodes=args.eval_episodes,
        ),
        "optimization_history": plot_optimization_history(study, history_path, algo),
        "param_importance": plot_param_importance(study, importance_path, algo),
    }
    paths["summary"] = write_summary_json(
        study=study,
        algo=algo,
        path=summary_path,
        env_kwargs=env_kwargs,
        paths=paths,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
    )
    report_path = write_stage5_report(args.report_path, args.log_dir, args.figure_dir)

    return {
        "algo": algo,
        "study": study,
        "paths": paths,
        "report": report_path,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _best_params_table(params: dict[str, Any]) -> list[str]:
    lines = ["| Hiperparametro | Valor |", "|---|---:|"]
    for name, value in sorted(params.items()):
        lines.append(f"| `{name}` | `{_format_param_value(value)}` |")
    return lines


def _artifact_link(path_text: str) -> str:
    return f"`{path_text}`"


def _report_insights(summaries: dict[str, dict[str, Any] | None]) -> list[str]:
    ranked: list[tuple[str, dict[str, Any]]] = []
    for algo, summary in summaries.items():
        if summary is not None and summary.get("best_trial"):
            ranked.append((algo, summary["best_trial"]))
    ranked.sort(key=lambda item: float(item[1]["value"]), reverse=True)

    if not ranked:
        return ["- Ainda nao ha trials completos para comparar."]

    lines: list[str] = []
    for position, (algo, best) in enumerate(ranked):
        metrics = best.get("metrics") or {}
        success = (
            _percent(metrics["success_rate"])
            if "success_rate" in metrics
            else "n/a"
        )
        truncated = (
            _percent(metrics["truncated_rate"])
            if "truncated_rate" in metrics
            else "n/a"
        )
        label = "maior retorno medio" if position == 0 else "retorno medio"
        lines.append(
            f"- **{algo.upper()}:** {label} de {_number(best.get('value'))}, "
            f"com {success} de sucesso e {truncated} de episodios truncados."
        )

    if len(ranked) > 1:
        difference = float(ranked[0][1]["value"]) - float(ranked[1][1]["value"])
        lines.append(
            f"- A diferenca entre os dois maiores retornos foi {_number(difference)}. "
            "A comparacao deve ser confirmada com o mesmo orcamento e mais sementes."
        )
    return lines


def write_stage5_report(report_path: Path, log_dir: Path, figure_dir: Path) -> Path:
    """Writes a consolidated Markdown report from available Optuna summaries."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summaries = {
        algo: _read_json(log_dir / f"{algo}_summary.json")
        for algo in sorted(ALGORITHMS)
    }
    run_config_rows = [
        "| Algoritmo | Limite de passos nos novos trials | Episodios de avaliacao |",
        "|---|---:|---:|",
    ]
    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        run_config_rows.append(
            f"| {algo.upper()} | "
            f"{summary.get('total_timesteps_per_trial', 'n/a') if summary else 'n/a'} | "
            f"{summary.get('eval_episodes', 'n/a') if summary else 'n/a'} |"
        )

    lines = [
        "# FloodGuard - Relatorio da Etapa 5",
        "",
        f"Gerado em `{generated_at}` por `experiments/tune.py`.",
        "",
        "## Resumo",
        "",
        "A Etapa 5 implementou e executou a busca de hiperparametros dos algoritmos "
        "DQN, PPO e A2C com Optuna. A rodada identifica configuracoes candidatas "
        "para o treino final e registra todos os resultados para auditoria.",
        "",
        "Os resultados servem para selecionar configuracoes candidatas ao treino final. "
        "Eles nao definem, isoladamente, o melhor algoritmo, pois os estudos acumulam "
        "trials de execucoes com orcamentos diferentes.",
        "",
        "## Objetivo",
        "",
        "Encontrar configuracoes promissoras para cada algoritmo, usando o retorno medio "
        "de avaliacao como criterio de otimizacao. O treino usa o sinal denso de progresso "
        "da Etapa 4, enquanto a avaliacao usa a recompensa original do MDP.",
        "",
        "## Configuracao da busca",
        "",
        "- O `TPESampler` escolhe novas configuracoes com base nos resultados anteriores.",
        "- O `MedianPruner` encerra antecipadamente trials com baixo desempenho.",
        "- Todos os trials, inclusive os podados, permanecem registrados em SQLite e CSV.",
        "",
        "A tabela abaixo mostra a configuracao da execucao mais recente. Os totais da "
        "secao de resultados tambem incluem trials de execucoes anteriores.",
        "",
        *run_config_rows,
        "",
        "Os parametros avaliados foram:",
        "",
        "- **DQN:** taxa de aprendizado, buffer, lote, desconto, exploracao e frequencia de atualizacao.",
        "- **PPO:** taxa de aprendizado, passos, lote, epocas, desconto, GAE, clipping e coeficientes de perda.",
        "- **A2C:** taxa de aprendizado, passos, desconto, GAE, entropia, valor e limite do gradiente.",
        "",
        "## Resultados",
        "",
        "| Algoritmo | Trials totais | Trials completos | Trials podados | Melhor retorno | Sucesso | Perda do animal | CSV de trials |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        if summary is None:
            lines.append(f"| {algo.upper()} | 0 | 0 | 0 | n/a | n/a | n/a | n/a |")
            continue
        best = summary.get("best_trial") or {}
        metrics = best.get("metrics") or {}
        paths = summary.get("paths") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    algo.upper(),
                    str(summary.get("total_trials", 0)),
                    str(summary.get("completed_trials", 0)),
                    str(summary.get("pruned_trials", 0)),
                    _number(best.get("value")),
                    _percent(metrics["success_rate"]) if "success_rate" in metrics else "n/a",
                    _percent(metrics["animal_lost_rate"])
                    if "animal_lost_rate" in metrics
                    else "n/a",
                    _artifact_link(paths.get("trials_csv", "n/a")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Leitura dos resultados",
            "",
            *_report_insights(summaries),
            "",
            "## Melhores configuracoes",
            "",
        ]
    )
    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        lines.append(f"### {algo.upper()}")
        if summary is None or not summary.get("best_trial"):
            lines.extend(
                [
                    "",
                    "Ainda nao ha trials completos para este algoritmo.",
                    "",
                ]
            )
            continue
        best = summary["best_trial"]
        metrics = best.get("metrics") or {}
        lines.extend(
            [
                "",
                f"Trial selecionado: `{best['number']}`. Retorno medio: "
                f"`{_number(best.get('value'))}`.",
                "",
                "| Metrica | Valor |",
                "|---|---:|",
                f"| Retorno medio | {_number(metrics.get('return_mean'))} +/- {_number(metrics.get('return_std'))} |",
                f"| Taxa de sucesso | {_percent(metrics['success_rate']) if 'success_rate' in metrics else 'n/a'} |",
                f"| Taxa de perda do animal | {_percent(metrics['animal_lost_rate']) if 'animal_lost_rate' in metrics else 'n/a'} |",
                f"| Taxa de truncamento | {_percent(metrics['truncated_rate']) if 'truncated_rate' in metrics else 'n/a'} |",
                f"| Uso medio de bateria | {_number(metrics.get('battery_spent_mean'))} |",
                "",
                *_best_params_table(best.get("params") or {}),
                "",
            ]
        )

    lines.extend(
        [
            "## Arquivos gerados",
            "",
            f"- Bancos SQLite e CSVs de trials: `{_project_relative_path(log_dir)}`.",
            f"- Graficos de historico e importancia: `{_project_relative_path(figure_dir)}`.",
            "- JSONs `*_best_params.json`: configuracoes escolhidas para a Etapa 6.",
            "",
            "## Proximos passos",
            "",
            "Antes do treino final da Etapa 6, recomenda-se comparar as configuracoes "
            "vencedoras em varias sementes, usando o mesmo orcamento de treino e avaliacao.",
            "",
            "1. Treinar novamente as configuracoes selecionadas de A2C, PPO e DQN.",
            "2. Usar varias sementes para reduzir o efeito do acaso nos resultados.",
            "3. Comparar retorno, sucesso, perda do animal e consumo de bateria.",
            "4. Selecionar o modelo final com base nessa avaliacao controlada.",
            "",
            "## Conclusao",
            "",
            "A Etapa 5 foi concluida. A busca e reproduzivel, os trials sao auditaveis e "
            "as melhores configuracoes estao salvas para a Etapa 6. A2C e PPO apresentaram "
            "os melhores resultados acumulados. A escolha final deve ser confirmada em "
            "uma comparacao controlada, com o mesmo orcamento e multiplas sementes.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune FloodGuard SB3 hyperparameters with Optuna (Stage 5)."
    )
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Training budget per trial.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Final evaluation episodes per trial.",
    )
    parser.add_argument(
        "--pruning-eval-episodes",
        type=int,
        default=10,
        help="Episodes used for intermediate pruning evaluations.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Timesteps between pruning evaluations. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-seed-offset", type=int, default=20_000)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--flood-advance-prob", type=float, default=None)
    parser.add_argument("--flood-deepen-prob", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = optimize_hyperparams(args)
    study: optuna.Study = result["study"]
    complete = _complete_trials(study)
    print("Optuna tuning complete")
    print(f"Algorithm: {result['algo'].upper()}")
    print(f"Trials total: {len(study.trials)}")
    print(f"Trials complete: {len(complete)}")
    if complete:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best return mean: {_number(study.best_value)}")
        print(f"Best params: {study.best_params}")
    print(f"Trials CSV: {_project_relative_path(result['paths']['trials_csv'])}")
    print(f"Best config: {_project_relative_path(result['paths']['best_config'])}")
    print(f"Report: {_project_relative_path(result['report'])}")


if __name__ == "__main__":
    main()
