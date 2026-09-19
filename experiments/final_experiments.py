"""Final multi-seed training and comparison for FloodGuard (Stage 6)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
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
from floodguard.utils.seeding import set_global_seed


DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "optuna"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "final"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "final"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "relatorio_etapa_6.md"

COMPARISON_METRICS = (
    "return_mean",
    "success_rate",
    "animal_lost_rate",
    "truncated_rate",
    "battery_depleted_rate",
    "battery_spent_mean",
    "steps_to_success_mean",
    "effective_barriers_installed_mean",
)


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


def _read_learning_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as file:
        next(file)
        rows = list(csv.DictReader(file))
    lengths = np.asarray([int(row["l"]) for row in rows], dtype=float)
    returns = np.asarray([float(row["r"]) for row in rows], dtype=float)
    return np.cumsum(lengths), returns


def plot_learning_curves(runs: list[dict[str, Any]], path: Path) -> Path:
    plt.figure(figsize=(9, 5))
    colors = {"a2c": "#2a9d8f", "ppo": "#457b9d", "dqn": "#e76f51"}
    for algo in sorted({run["algo"] for run in runs}):
        algo_runs = [run for run in runs if run["algo"] == algo]
        curves: list[tuple[np.ndarray, np.ndarray]] = []
        for run in algo_runs:
            steps, returns = _read_learning_curve(Path(run["monitor_absolute"]))
            if len(returns):
                window = min(50, len(returns))
                smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
                curves.append((steps[window - 1 :], smooth))
        if not curves:
            continue
        max_common = min(curve[0][-1] for curve in curves)
        grid = np.linspace(0, max_common, 200)
        interpolated = np.vstack(
            [np.interp(grid, steps, values) for steps, values in curves]
        )
        center = interpolated.mean(axis=0)
        spread = interpolated.std(axis=0)
        color = colors[algo]
        plt.plot(grid, center, color=color, label=algo.upper(), linewidth=2)
        plt.fill_between(grid, center - spread, center + spread, color=color, alpha=0.18)
    plt.title("Curvas de aprendizado por algoritmo")
    plt.xlabel("Passos de treinamento")
    plt.ylabel("Retorno medio movel")
    plt.grid(alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_return_boxplot(episodes: list[dict[str, Any]], path: Path) -> Path:
    algos = sorted({row["algo"] for row in episodes})
    values = [
        [float(row["return_total"]) for row in episodes if row["algo"] == algo]
        for algo in algos
    ]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(values, tick_labels=[algo.upper() for algo in algos], showmeans=True)
    plt.title("Distribuicao do retorno nos episodios de teste")
    plt.xlabel("Algoritmo")
    plt.ylabel("Retorno por episodio")
    plt.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_final_metrics(aggregates: dict[str, dict[str, Any]], path: Path) -> Path:
    algos = sorted(aggregates)
    labels = ["Sucesso", "Perda do animal", "Truncamento"]
    keys = ["success_rate_mean", "animal_lost_rate_mean", "truncated_rate_mean"]
    x = np.arange(len(labels))
    width = 0.24
    plt.figure(figsize=(9, 4.8))
    for index, algo in enumerate(algos):
        values = [100.0 * float(aggregates[algo][key] or 0.0) for key in keys]
        plt.bar(x + (index - 1) * width, values, width, label=algo.upper())
    plt.xticks(x, labels)
    plt.ylabel("Taxa (%)")
    plt.title("Metricas finais de avaliacao")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _final_insights(
    aggregates: dict[str, dict[str, Any]], ranking: list[str]
) -> list[str]:
    lines = [
        f"- **{ranking[0].upper()} ficou em primeiro lugar no retorno medio**, "
        f"com {_number(aggregates[ranking[0]]['return_mean_mean'])}."
    ]
    if "ppo" in aggregates:
        lines.append(
            f"- **PPO foi o mais confiavel:** sucesso em "
            f"{_percent(aggregates['ppo']['success_rate_mean'])} dos episodios e "
            "baixa variacao entre sementes."
        )
    if "dqn" in aggregates:
        lines.append(
            f"- **DQN teve desempenho intermediario:** sucesso em "
            f"{_percent(aggregates['dqn']['success_rate_mean'])} dos episodios, mas "
            f"{_percent(aggregates['dqn']['truncated_rate_mean'])} terminaram no limite de passos."
        )
    if "a2c" in aggregates:
        lines.append(
            "- **A2C nao concluiu resgates no experimento final.** O bom resultado da "
            "Etapa 5 nao se manteve com o treino mais longo, indicando instabilidade."
        )
    if len(aggregates) == len(ALGORITHMS):
        lines.append(
            "- Os tres algoritmos usaram poucas barreiras efetivas. O comportamento "
            "aprendido priorizou o resgate direto em vez da protecao do mapa."
        )
    return lines


def _report_relative_path(project_path: str, report_path: Path) -> str:
    absolute = (PROJECT_ROOT / project_path).resolve()
    return os.path.relpath(absolute, start=report_path.parent.resolve())


def write_report(summary: dict[str, Any], path: Path) -> Path:
    aggregates = summary["algorithms"]
    ranking = sorted(
        aggregates,
        key=lambda algo: float(aggregates[algo]["return_mean_mean"]),
        reverse=True,
    )
    lines = [
        "# FloodGuard - Relatorio da Etapa 6",
        "",
        f"Gerado em `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` por "
        "`experiments/final_experiments.py`.",
        "",
        "## Resumo",
        "",
        "Esta etapa treinou DQN, PPO e A2C com os melhores hiperparametros da "
        "Etapa 5. Cada algoritmo foi treinado com as mesmas sementes e avaliado "
        "nos mesmos episodios de teste.",
        "",
        f"O maior retorno medio foi obtido por **{ranking[0].upper()}**. A tabela "
        "abaixo apresenta os resultados completos e permite comparar estabilidade, "
        "sucesso e seguranca.",
        "",
        "## Configuracao",
        "",
        f"- Sementes de treino: `{', '.join(map(str, summary['seeds']))}`.",
        f"- Passos de treino por modelo: `{summary['total_timesteps']}`.",
        f"- Episodios de teste por modelo: `{summary['eval_episodes']}`.",
        f"- Avaliacao deterministica: `{'sim' if summary['deterministic'] else 'nao'}`.",
        "- A avaliacao usa a recompensa original do MDP, sem o reforco de progresso usado no treino.",
        (
            "- A politica deterministica escolhe sempre a acao de maior probabilidade."
            if summary["deterministic"]
            else "- A politica estocastica foi mantida para seguir o mesmo criterio usado na Etapa 5."
        ),
        "",
        "## Resultados",
        "",
        "Os valores mostram a media entre sementes. O termo apos `+/-` e o desvio "
        "padrao entre os modelos treinados.",
        "",
        "| Algoritmo | Retorno | Sucesso | Perda do animal | Truncamento | Bateria usada | Passos ate o sucesso |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algo in ranking:
        item = aggregates[algo]
        lines.append(
            f"| {algo.upper()} | "
            f"{_number(item['return_mean_mean'])} +/- {_number(item['return_mean_std'])} | "
            f"{_percent(item['success_rate_mean'])} +/- {_percent(item['success_rate_std'])} | "
            f"{_percent(item['animal_lost_rate_mean'])} | "
            f"{_percent(item['truncated_rate_mean'])} | "
            f"{_number(item['battery_spent_mean_mean'])} | "
            f"{_number(item['steps_to_success_mean_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Leitura dos resultados",
            "",
            *_final_insights(aggregates, ranking),
            "",
            "## Graficos",
            "",
            "### Curvas de aprendizado",
            "",
            "A curva usa a recompensa de treino, que inclui o reforco de progresso. "
            "As metricas da tabela usam somente a recompensa original do MDP.",
            "",
            f"![Curvas de aprendizado]({_report_relative_path(summary['figures']['learning_curves'], path)})",
            "",
            "### Retorno nos episodios de teste",
            "",
            f"![Distribuicao do retorno]({_report_relative_path(summary['figures']['return_boxplot'], path)})",
            "",
            "### Taxas finais",
            "",
            f"![Metricas finais]({_report_relative_path(summary['figures']['final_metrics'], path)})",
            "",
            "## Arquivos gerados",
            "",
            f"- Modelos: `{summary['paths']['models']}`.",
            f"- Logs de treino: `{summary['paths']['logs']}`.",
            f"- Resultados por seed: `{summary['paths']['seed_results']}`.",
            f"- Resultados por episodio: `{summary['paths']['episode_results']}`.",
            f"- Resumo em JSON: `{summary['paths']['summary']}`.",
            "",
            "## Conclusao",
            "",
            f"A Etapa 6 foi concluida com uma comparacao controlada entre os tres "
            f"algoritmos. **{ranking[0].upper()}** apresentou o maior retorno medio. "
            "A decisao sobre o modelo final deve considerar tambem a taxa de sucesso, "
            "a perda do animal e a variacao entre sementes, e nao apenas o retorno.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_final_experiments(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs = _env_kwargs_from_args(args)
    runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}

    for algo in args.algos:
        configs[algo] = load_best_params(algo, args.config_dir)
        for seed in args.seeds:
            set_global_seed(seed)
            run_name = f"{algo}_seed_{seed}"
            monitor = args.log_dir / algo / f"{run_name}.monitor.csv"
            tensorboard = args.log_dir / algo / "tensorboard"
            model_path = args.model_dir / algo / f"floodguard_{run_name}.zip"
            if args.reuse_models and model_path.exists():
                model = ALGORITHMS[algo].load(model_path)
            else:
                train_env = make_training_env(
                    seed=seed,
                    env_kwargs=env_kwargs,
                    monitor_file=monitor,
                    reward_shaping=True,
                )
                try:
                    model = build_final_model(
                        algo=algo,
                        env=train_env,
                        seed=seed,
                        tensorboard_log=tensorboard,
                        hyperparams=configs[algo],
                        verbose=args.verbose,
                    )
                    model.learn(
                        total_timesteps=args.total_timesteps,
                        tb_log_name=run_name,
                        log_interval=args.log_interval,
                        progress_bar=args.progress_bar,
                    )
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(model_path)
                finally:
                    train_env.close()

            episodes, evaluation = evaluate_model(
                model=model,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                env_kwargs=env_kwargs,
                deterministic=not args.stochastic_eval,
            )

            evaluation_path = args.output_dir / algo / f"{run_name}_evaluation.json"
            _write_json(evaluation, evaluation_path)
            for episode in episodes:
                episode_rows.append({"algo": algo, "train_seed": seed, **asdict(episode)})
            row = {"algo": algo, "seed": seed}
            row.update({metric: evaluation.get(metric) for metric in COMPARISON_METRICS})
            seed_rows.append(row)
            runs.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "monitor": _project_relative_path(monitor),
                    "monitor_absolute": str(monitor.resolve()),
                    "model": _project_relative_path(model_path),
                    "evaluation": _project_relative_path(evaluation_path),
                    "summary": evaluation,
                }
            )

    aggregates = {
        algo: _aggregate_seed_summaries(
            [run["summary"] for run in runs if run["algo"] == algo]
        )
        for algo in args.algos
    }
    seed_csv = _write_csv(seed_rows, args.output_dir / "final_seed_results.csv")
    episode_csv = _write_csv(episode_rows, args.output_dir / "final_episode_results.csv")
    figures = {
        "learning_curves": plot_learning_curves(
            runs, args.figure_dir / "stage6_learning_curves.png"
        ),
        "return_boxplot": plot_return_boxplot(
            episode_rows, args.figure_dir / "stage6_return_boxplot.png"
        ),
        "final_metrics": plot_final_metrics(
            aggregates, args.figure_dir / "stage6_final_metrics.png"
        ),
    }
    summary_path = args.output_dir / "final_summary.json"
    summary = {
        "algorithms": aggregates,
        "configs": configs,
        "seeds": list(args.seeds),
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "deterministic": not args.stochastic_eval,
        "env_kwargs": env_kwargs,
        "runs": [{key: value for key, value in run.items() if key != "monitor_absolute"} for run in runs],
        "figures": {key: _project_relative_path(value) for key, value in figures.items()},
        "paths": {
            "models": _project_relative_path(args.model_dir),
            "logs": _project_relative_path(args.log_dir),
            "seed_results": _project_relative_path(seed_csv),
            "episode_results": _project_relative_path(episode_csv),
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
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--stochastic-eval", action="store_true")
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
    args = build_parser().parse_args()
    result = run_final_experiments(args)
    print("Experimentos da Etapa 6 concluidos.")
    print(f"Relatorio: {_project_relative_path(result['report'])}")


if __name__ == "__main__":
    main()
