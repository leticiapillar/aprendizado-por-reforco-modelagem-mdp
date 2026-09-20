"""Smoke tests for Stage 6 final experiments."""

import json
from argparse import Namespace

from experiments.final_experiments import run_final_experiments


def test_stage6_writes_comparison_artifacts(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "a2c_best_params.json").write_text(
        json.dumps(
            {
                "algo": "a2c",
                "best_params": {
                    "learning_rate": 0.001,
                    "n_steps": 5,
                    "gamma": 0.95,
                    "gae_lambda": 0.9,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        algos=["a2c"],
        seeds=[1, 2],
        total_timesteps=2,
        eval_episodes=2,
        eval_seed=100,
        config_dir=config_dir,
        log_dir=tmp_path / "logs",
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "final",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "relatorio_etapa_6.md",
        curve_eval_freq=1,
        curve_eval_episodes=1,
        workers=1,
        report_only=False,
        with_defaults=True,
        ablations=False,
        render_episodes=True,
        reuse_models=False,
        progress_bar=False,
        verbose=0,
        log_interval=1,
        max_steps=2,
        flood_advance_prob=0.0,
        flood_deepen_prob=0.0,
    )

    result = run_final_experiments(args)

    summary = result["summary"]
    assert result["report"].exists()
    for name in ("final_summary.json", "final_seed_results.csv", "final_episode_results.csv",
                 "learning_curves.csv", "eval_curves.csv"):
        assert (args.output_dir / name).exists()
    for name in ("stage6_learning_curves", "stage6_eval_curves", "stage6_return_boxplot",
                 "stage6_final_metrics", "stage6_default_vs_tuned", "stage6_episode_a2c"):
        assert list(args.figure_dir.glob(f"{name}.*"))
    assert summary["primary_eval_mode"] == "deterministic"
    assert set(summary["aggregates"]) == {"tuned", "default"}
    assert set(summary["aggregates"]["tuned"]["a2c"]) == {"deterministic", "stochastic"}
    assert "stall_step_rate_mean" in summary["aggregates"]["tuned"]["a2c"]["deterministic"]
    assert "avaliacao principal" in result["report"].read_text(encoding="utf-8").lower()

    assert "Referencia do MDP" in result["report"].read_text(encoding="utf-8")

    # `--report-only` reconstroi figuras e relatorio dos arquivos salvos, sem treinar.
    args.report_only = True
    args.render_episodes = False
    args.report_path = tmp_path / "relatorio_regenerado.md"
    regenerated = run_final_experiments(args)
    assert regenerated["report"].exists()
    assert regenerated["summary"]["aggregates"] == summary["aggregates"]
