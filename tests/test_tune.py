"""Smoke tests for the Stage 5 Optuna tuning script."""

from argparse import Namespace
import json

import optuna

import pytest

from experiments.tune import DEFAULT_PARAMS, optimize_hyperparams, suggest_hyperparams


def test_stage5_search_spaces_cover_required_algorithms():
    expected_keys = {
        "dqn": {
            "learning_rate",
            "buffer_size",
            "batch_size",
            "gamma",
            "exploration_fraction",
            "exploration_final_eps",
            "target_update_interval",
            "train_freq",
            "gradient_steps",
            "learning_starts",
        },
        "ppo": {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
        },
        "a2c": {
            "learning_rate",
            "n_steps",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        },
    }

    for algo, keys in expected_keys.items():
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparams(trial, algo)
        assert set(params) == keys


def _tiny_args(tmp_path, **overrides):
    args = Namespace(
        algo="a2c",
        n_trials=1,
        timeout=None,
        total_timesteps=2,
        eval_episodes=1,
        pruning_eval_episodes=1,
        eval_freq=0,
        seed=123,
        eval_seed_offset=100,
        n_startup_trials=1,
        pruner_warmup_evals=2,
        sampler_seed=None,
        enqueue_default=True,
        stochastic_eval=False,
        log_dir=tmp_path / "logs",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "relatorio_etapa_5.md",
        log_interval=1,
        progress_bar=False,
        verbose=0,
        max_steps=2,
        flood_advance_prob=0.0,
        flood_deepen_prob=0.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_default_params_lie_inside_search_space():
    for algo, defaults in DEFAULT_PARAMS.items():
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial(defaults)
        trial = study.ask()
        assert suggest_hyperparams(trial, algo) == defaults


def test_existing_study_with_other_budget_is_rejected(tmp_path):
    optimize_hyperparams(_tiny_args(tmp_path))
    with pytest.raises(ValueError, match="orcamento"):
        optimize_hyperparams(_tiny_args(tmp_path, total_timesteps=4))


def test_optimize_hyperparams_writes_stage5_artifacts(tmp_path):
    args = _tiny_args(tmp_path)

    result = optimize_hyperparams(args)

    assert result["algo"] == "a2c"
    assert result["paths"]["trials_csv"].exists()
    assert result["paths"]["best_config"].exists()
    assert result["paths"]["summary"].exists()
    assert result["paths"]["optimization_history"].exists()
    assert result["paths"]["param_importance"].exists()
    assert result["report"].exists()
    assert "A2C" in result["report"].read_text(encoding="utf-8")

    summary = json.loads(result["paths"]["summary"].read_text(encoding="utf-8"))
    assert not summary["storage"].startswith("/")
    assert all(not path.startswith("/") for path in summary["paths"].values())

    assert summary["default_trial"]["params"] == DEFAULT_PARAMS["a2c"]
    assert summary["budget"]["eval_mode"] == "deterministic"
    assert len(summary["trials"]) == 1
    assert result["paths"]["intermediate_csv"].exists()
