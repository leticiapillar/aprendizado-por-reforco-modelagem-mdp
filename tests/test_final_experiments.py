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
        stochastic_eval=False,
        reuse_models=False,
        progress_bar=False,
        verbose=0,
        log_interval=1,
        max_steps=2,
        flood_advance_prob=0.0,
        flood_deepen_prob=0.0,
    )

    result = run_final_experiments(args)

    assert result["report"].exists()
    assert (args.output_dir / "final_summary.json").exists()
    assert (args.output_dir / "final_seed_results.csv").exists()
    assert (args.output_dir / "final_episode_results.csv").exists()
    assert (args.figure_dir / "stage6_learning_curves.png").exists()
    assert (args.figure_dir / "stage6_return_boxplot.png").exists()
    assert (args.figure_dir / "stage6_final_metrics.png").exists()
    assert result["summary"]["deterministic"] is True
