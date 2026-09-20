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
import numpy as np
import optuna
import torch
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


# Hiperparametros default do SB3 (dentro do espaco de busca). Entram como o
# trial 0 de cada estudo, com o mesmo orcamento e as mesmas seeds dos demais,
# para que o ganho da otimizacao seja medido contra o default.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.10,
        "exploration_final_eps": 0.05,
        "target_update_interval": 10_000,
        "train_freq": 4,
        "gradient_steps": 1,
        "learning_starts": 100,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2_048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 1e-8,
        "vf_coef": 0.5,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 1e-8,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}


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
                "target_update_interval", [250, 500, 1_000, 2_000, 5_000, 10_000]
            ),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
            "learning_starts": trial.suggest_categorical(
                "learning_starts", [100, 500, 1_000, 2_000]
            ),
        }

    if algo == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1_024, 2_048])
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
        deterministic: bool = True,
    ):
        super().__init__(verbose=0)
        self.deterministic = deterministic
        self.trial = trial
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.env_kwargs = env_kwargs
        self.total_timesteps = total_timesteps
        self.reports: list[float] = []

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        _episodes, summary = evaluate_model(
            model=self.model,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
            env_kwargs=self.env_kwargs,
            deterministic=self.deterministic,
        )
        mean_return = float(summary["return_mean"])
        self.reports.append(mean_return)
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

    plt.plot(numbers, values, marker="o", label="Score do trial")
    plt.plot(numbers, best_so_far, marker="s", label="Melhor ate aqui")
    plt.title(f"Optuna - historico de otimizacao ({algo.upper()})")
    plt.xlabel("Trial")
    plt.ylabel("Score (retorno medio de avaliacao ao longo do treino)")
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


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    """Uma linha por trial (inclusive podados) para o relatorio."""
    rows = []
    for trial in study.trials:
        last_step = max(trial.intermediate_values) if trial.intermediate_values else None
        value = trial.value
        if value is None and last_step is not None:
            value = trial.intermediate_values[last_step]
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": value,
                "final": trial.value is not None,
                "last_step": last_step,
                "success_rate": trial.user_attrs.get("success_rate"),
                "params": trial.params,
            }
        )
    return rows


def write_intermediate_csv(study: optuna.Study, path: Path) -> Path:
    """Retornos intermediarios (usados pelo pruner) de todos os trials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["trial", "state", "step", "return_mean"])
        for trial in study.trials:
            for step, value in sorted(trial.intermediate_values.items()):
                writer.writerow([trial.number, trial.state.name, step, value])
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
        "budget": study.user_attrs.get("budget"),
        "trials": _trial_rows(study),
        "paths": {
            key: _project_relative_path(value)
            for key, value in paths.items()
            if key != "storage_url"
        },
    }
    default_trial = study.trials[0] if study.trials else None
    if default_trial is not None and default_trial.params == DEFAULT_PARAMS.get(algo):
        payload["default_trial"] = {
            "number": default_trial.number,
            "state": default_trial.state.name,
            "value": default_trial.value,
            "params": default_trial.params,
            "metrics": {
                key: default_trial.user_attrs.get(key)
                for key in REPORT_METRICS
                if key in default_trial.user_attrs
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
    warmup_steps: int = 0,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=warmup_steps,
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


def _budget(args: argparse.Namespace, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "pruning_eval_episodes": args.pruning_eval_episodes,
        "eval_freq": args.eval_freq,
        "seed": args.seed,
        "eval_seed_offset": args.eval_seed_offset,
        "eval_mode": "stochastic" if args.stochastic_eval else "deterministic",
        "env_kwargs": env_kwargs,
    }


def _check_budget(
    study: optuna.Study, args: argparse.Namespace, env_kwargs: dict[str, Any]
) -> None:
    """Impede misturar trials de orcamentos/protocolos diferentes no mesmo estudo."""
    budget = _budget(args, env_kwargs)
    stored = study.user_attrs.get("budget")
    if stored is None:
        study.set_user_attr("budget", budget)
    elif stored != budget:
        raise ValueError(
            "O estudo existente foi criado com outro orcamento/protocolo "
            f"({stored}). Use outro --log-dir ou apague o banco para recomecar."
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
    # Seed de treino e episodios de avaliacao identicos em todos os trials: a
    # diferenca de retorno vem dos hiperparametros, nao do sorteio de seeds.
    trial_seed = args.seed
    eval_seed = args.seed + args.eval_seed_offset
    deterministic = not args.stochastic_eval
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
            eval_seed=eval_seed,
            env_kwargs=env_kwargs,
            total_timesteps=args.total_timesteps,
            deterministic=deterministic,
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
            seed=eval_seed,
            env_kwargs=env_kwargs,
            deterministic=deterministic,
        )
        _set_trial_metrics(trial, summary)
        trial.set_user_attr("final_return_mean", float(summary["return_mean"]))
        # Objetivo: retorno medio das avaliacoes ao longo do treino parcial (area
        # sob a curva de aprendizado). Como o cenario inicial e fixo, muitas
        # configuracoes atingem o retorno otimo no fim; a media ao longo do treino
        # as desempata pela velocidade de convergencia.
        score = float(np.mean(callback.reports)) if callback.reports else float(summary["return_mean"])
        trial.set_user_attr("learning_curve_score", score)
        for key in ("stall_step_rate", "stalled_episode_rate"):
            trial.set_user_attr(key, summary.get(key))
        trial.set_user_attr("total_timesteps", args.total_timesteps)
        trial.set_user_attr("train_seed", trial_seed)
        trial.set_user_attr("eval_mode", "deterministic" if deterministic else "stochastic")
        trial.set_user_attr("monitor", _project_relative_path(monitor_file))
        trial.set_user_attr("tensorboard", _project_relative_path(tensorboard_dir))
        return score
    finally:
        train_env.close()


def optimize_hyperparams(args: argparse.Namespace) -> dict[str, Any]:
    algo = args.algo.lower()
    seed = set_global_seed(args.seed)
    env_kwargs = _env_kwargs_from_args(args)
    # Processos que compartilham o mesmo banco precisam de seeds de sampler
    # distintas; caso contrario sorteiam configuracoes identicas.
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else seed

    storage_path = args.log_dir / f"optuna_{algo}.db"
    trials_csv_path = args.log_dir / f"{algo}_trials.csv"
    best_config_path = args.log_dir / f"{algo}_best_params.json"
    summary_path = args.log_dir / f"{algo}_summary.json"
    history_path = args.figure_dir / f"optuna_{algo}_history.png"
    importance_path = args.figure_dir / f"optuna_{algo}_param_importance.png"

    warmup_steps = args.pruner_warmup_evals * max(args.eval_freq, 0)
    study = _make_study(
        algo=algo,
        storage_path=storage_path,
        seed=sampler_seed,
        n_startup_trials=args.n_startup_trials,
        warmup_steps=warmup_steps,
    )
    _check_budget(study, args, env_kwargs)
    if not study.trials and args.enqueue_default:
        study.enqueue_trial(DEFAULT_PARAMS[algo])
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
        "intermediate_csv": write_intermediate_csv(
            study, args.log_dir / f"{algo}_intermediate.csv"
        ),
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


def _compact_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{name}={_format_param_value(value)}" for name, value in sorted(params.items()))


def _metric(metrics: dict[str, Any], key: str, percent: bool = False) -> str:
    if metrics.get(key) is None:
        return "n/a"
    return _percent(metrics[key]) if percent else _number(metrics[key])


def _report_insights(summaries: dict[str, dict[str, Any] | None]) -> list[str]:
    lines: list[str] = []
    for algo, summary in summaries.items():
        if not summary or not summary.get("best_trial"):
            continue
        best = summary["best_trial"]
        default = summary.get("default_trial") or {}
        metrics = best.get("metrics") or {}
        text = (
            f"- **{algo.upper()}:** melhor score {_number(best.get('value'))} "
            f"(trial {best['number']}, {_metric(metrics, 'success_rate', True)} de sucesso)"
        )
        if default.get("value") is not None:
            gain = float(best["value"]) - float(default["value"])
            text += (
                f"; o default do SB3 obteve {_number(default['value'])} "
                f"(ganho de {_number(gain)})."
            )
            if best["number"] == default["number"]:
                text += " O default foi o melhor trial: a busca nao superou o default."
        else:
            text += "."
        lines.append(text)
    if not lines:
        return ["- Ainda nao ha trials completos para comparar."]
    lines.append(
        "- Todos os trials usam a mesma seed de treino, os mesmos episodios de "
        "avaliacao e o mesmo orcamento; diferencas pequenas de retorno estao "
        "dentro do ruido de 1 seed e devem ser confirmadas na Etapa 6 com varias seeds."
    )
    return lines


def _trial_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Trial | Estado | Score | Sucesso final | Hiperparametros |",
        "|---:|---|---:|---:|---|",
    ]
    for row in summary.get("trials") or []:
        value = _number(row["value"])
        if not row["final"] and row["value"] is not None:
            value += f" (passo {row['last_step']})"
        success = "n/a" if row.get("success_rate") is None else _percent(row["success_rate"])
        label = "DEFAULT" if row["number"] == 0 and summary.get("default_trial") else row["state"]
        lines.append(
            f"| {row['number']} | {label} | {value} | {success} | "
            f"{_compact_params(row['params'])} |"
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
        "| Algoritmo | Passos por trial | Episodios de avaliacao final | Episodios de avaliacao do pruning | Avaliacao |",
        "|---|---:|---:|---:|---|",
    ]
    for algo in sorted(ALGORITHMS):
        budget = (summaries[algo] or {}).get("budget") or {}
        run_config_rows.append(
            f"| {algo.upper()} | {budget.get('total_timesteps', 'n/a')} | "
            f"{budget.get('eval_episodes', 'n/a')} | "
            f"{budget.get('pruning_eval_episodes', 'n/a')} | "
            f"{budget.get('eval_mode', 'n/a')} |"
        )

    lines = [
        "# FloodGuard - Relatorio da Etapa 5",
        "",
        f"Gerado em `{generated_at}` por `experiments/tune.py`.",
        "",
        "## Resumo",
        "",
        "A Etapa 5 executou a busca de hiperparametros de DQN, PPO e A2C com Optuna. "
        "Cada estudo tem o mesmo orcamento em todos os trials, avalia a politica "
        "**deterministica** e inclui os hiperparametros default do SB3 como trial 0, "
        "de modo que o ganho da otimizacao e medido contra o default.",
        "",
        "## Objetivo",
        "",
        "Encontrar configuracoes promissoras para cada algoritmo. O criterio e o retorno "
        "medio de avaliacao **ao longo do treino parcial** (area sob a curva de aprendizado): "
        "como o cenario inicial e fixo, varias configuracoes atingem o retorno otimo ao "
        "final do treino, e a media ao longo do treino as desempata pela velocidade de "
        "convergencia. O retorno da avaliacao final tambem e registrado (`final_return_mean`). O treino usa o sinal denso de progresso da Etapa 4; "
        "a avaliacao usa a recompensa original do MDP.",
        "",
        "## Protocolo da busca",
        "",
        "- `TPESampler` propoe novas configuracoes com base nos resultados anteriores.",
        "- `MedianPruner` encerra trials abaixo da mediana, somente apos um periodo de "
        "aquecimento (`--pruner-warmup-evals` avaliacoes intermediarias).",
        "- **Seed de treino e episodios de avaliacao fixos** em todos os trials, para que "
        "a diferenca entre trials venha dos hiperparametros.",
        "- **Avaliacao deterministica**: acao de maior valor/probabilidade. No DQN a "
        "avaliacao estocastica seria epsilon-greedy com o `exploration_final_eps` do "
        "proprio modelo, o que faria a busca otimizar o modo de avaliacao.",
        "- O banco SQLite guarda o orcamento do estudo; tentar reutiliza-lo com outro "
        "orcamento gera erro, evitando misturar trials incomparaveis.",
        "- Todos os trials, inclusive os podados, ficam no SQLite, em `*_trials.csv` e "
        "(retornos intermediarios) em `*_intermediate.csv`.",
        "",
        *run_config_rows,
        "",
        "## Resultados",
        "",
        "| Algoritmo | Trials | Completos | Podados | Score do default | Melhor score | Retorno final (melhor) | Sucesso (melhor) | Perda do animal (melhor) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        if summary is None:
            lines.append(f"| {algo.upper()} | 0 | 0 | 0 | n/a | n/a | n/a | n/a |")
            continue
        best = summary.get("best_trial") or {}
        metrics = best.get("metrics") or {}
        default = summary.get("default_trial") or {}
        lines.append(
            f"| {algo.upper()} | {summary.get('total_trials', 0)} | "
            f"{summary.get('completed_trials', 0)} | {summary.get('pruned_trials', 0)} | "
            f"{_number(default.get('value'))} | {_number(best.get('value'))} | "
            f"{_metric(metrics, 'return_mean')} | "
            f"{_metric(metrics, 'success_rate', True)} | "
            f"{_metric(metrics, 'animal_lost_rate', True)} |"
        )

    lines.extend(["", "## Leitura dos resultados", "", *_report_insights(summaries), ""])
    lines.extend(["## Graficos", ""])
    for algo in sorted(ALGORITHMS):
        history = f"optuna_{algo}_history.png"
        importance = f"optuna_{algo}_param_importance.png"
        rel = os.path.relpath(figure_dir.resolve(), start=report_path.parent.resolve())
        lines.extend(
            [
                f"### {algo.upper()}",
                "",
                f"![Historico {algo.upper()}]({rel}/{history})",
                "",
                f"![Importancia {algo.upper()}]({rel}/{importance})",
                "",
            ]
        )

    lines.extend(["## Melhores configuracoes", ""])
    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        lines.append(f"### {algo.upper()}")
        if summary is None or not summary.get("best_trial"):
            lines.extend(["", "Ainda nao ha trials completos para este algoritmo.", ""])
            continue
        best = summary["best_trial"]
        metrics = best.get("metrics") or {}
        lines.extend(
            [
                "",
                f"Trial selecionado: `{best['number']}`. Score: "
                f"`{_number(best.get('value'))}`.",
                "",
                "| Metrica | Valor |",
                "|---|---:|",
                f"| Retorno medio | {_number(metrics.get('return_mean'))} +/- {_number(metrics.get('return_std'))} |",
                f"| Taxa de sucesso | {_metric(metrics, 'success_rate', True)} |",
                f"| Taxa de perda do animal | {_metric(metrics, 'animal_lost_rate', True)} |",
                f"| Taxa de truncamento | {_metric(metrics, 'truncated_rate', True)} |",
                f"| Uso medio de bateria | {_number(metrics.get('battery_spent_mean'))} |",
                "",
                *_best_params_table(best.get("params") or {}),
                "",
            ]
        )

    lines.extend(
        [
            "## Todas as configuracoes testadas",
            "",
            "Score = media dos retornos de avaliacao ao longo do treino. Trials podados mostram o ultimo retorno intermediario (com o passo em que "
            "foram encerrados). O trial `DEFAULT` usa os hiperparametros padrao do SB3.",
            "",
        ]
    )
    for algo in sorted(ALGORITHMS):
        summary = summaries[algo]
        lines.extend([f"### {algo.upper()}", ""])
        if summary is None:
            lines.extend(["Sem dados.", ""])
            continue
        lines.extend([*_trial_table(summary), ""])

    lines.extend(
        [
            "## Arquivos gerados",
            "",
            f"- Bancos SQLite, CSVs de trials e retornos intermediarios: `{_project_relative_path(log_dir)}`.",
            f"- Graficos de historico e importancia: `{_project_relative_path(figure_dir)}`.",
            "- JSONs `*_best_params.json`: configuracoes candidatas para a Etapa 6.",
            "",
            "## Limitacoes",
            "",
            "- Cada trial usa uma unica seed de treino: o retorno de um trial e uma "
            "estimativa ruidosa. A Etapa 6 reavalia as configuracoes com varias seeds.",
            "- O orcamento por trial e menor que o do treino final.",
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
    parser.add_argument("--n-trials", type=int, default=30)
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
        default=50,
        help="Final evaluation episodes per trial.",
    )
    parser.add_argument(
        "--pruning-eval-episodes",
        type=int,
        default=20,
        help="Episodes used for intermediate pruning evaluations.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Timesteps between (pruning/score) evaluations. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-seed-offset", type=int, default=20_000)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="TPE sampler seed (default: --seed). Use a different value per process "
        "when several processes share the same study.",
    )
    parser.add_argument(
        "--pruner-warmup-evals",
        type=int,
        default=2,
        help="Number of pruning evaluations before the MedianPruner may prune.",
    )
    parser.add_argument(
        "--no-enqueue-default",
        dest="enqueue_default",
        action="store_false",
        help="Do not run the SB3 default hyperparameters as trial 0.",
    )
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Evaluate with a sampled/epsilon-greedy policy (default: deterministic).",
    )
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
    # Redes MLP minusculas: varias threads de torch so causam contencao.
    torch.set_num_threads(1)
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
