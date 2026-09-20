"""Smoke tests for the Stage 4 stable-baselines3 integration."""

import gymnasium as gym
import numpy as np

from experiments.train import ALGORITHMS, build_model, evaluate_model, make_training_env


def test_stage4_declares_required_algorithms():
    assert set(ALGORITHMS) == {"dqn", "ppo", "a2c"}


def test_make_training_env_flattens_dict_observation():
    env = make_training_env(
        seed=123,
        env_kwargs={
            "max_steps": 5,
            "flood_advance_prob": 0.0,
            "flood_deepen_prob": 0.0,
        },
    )

    try:
        obs, info = env.reset(seed=123)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.contains(obs)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 1.0
        assert info["termination_reason"] == "running"
    finally:
        env.close()


def test_observation_is_float_and_keeps_position_information():
    env = make_training_env(seed=1, env_kwargs={"flood_advance_prob": 0.0, "flood_deepen_prob": 0.0})
    try:
        first, _ = env.reset(seed=1)
        second, *_ = env.step(0)  # UP: muda a posicao e a bateria

        assert env.observation_space.dtype == np.float32
        assert first.dtype == np.float32
        assert env.observation_space.contains(first)
        # Regressao: o RescaleObservation truncava a observacao para inteiros.
        fractional = first[(first > 0) & (first < 1)]
        assert fractional.size > 0
        assert np.count_nonzero(second - first) >= 2
    finally:
        env.close()


def test_normalize_observation_handles_zero_span_axes():
    env = make_training_env(seed=1, env_kwargs={"initial_barrier_kits": 0})
    try:
        obs, _ = env.reset(seed=1)
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_training_reward_reports_progress_toward_animal():
    env = make_training_env(
        seed=123,
        env_kwargs={"flood_advance_prob": 0.0, "flood_deepen_prob": 0.0},
        reward_shaping=True,
    )

    try:
        env.reset(seed=123)
        _obs, reward, _terminated, _truncated, info = env.step(0)  # UP

        assert info["progress_reward"] == 5.0
        assert reward == -1.0 + 5.0
    finally:
        env.close()


def test_evaluate_model_returns_stage3_metrics(tmp_path):
    env_kwargs = {
        "max_steps": 3,
        "flood_advance_prob": 0.0,
        "flood_deepen_prob": 0.0,
    }
    env = make_training_env(seed=123, env_kwargs=env_kwargs)

    try:
        model = build_model(
            algo="a2c",
            env=env,
            seed=123,
            tensorboard_log=tmp_path / "tensorboard",
        )
        episodes, summary = evaluate_model(
            model=model,
            episodes=2,
            seed=456,
            env_kwargs=env_kwargs,
        )

        assert len(episodes) == 2
        assert summary["episodes"] == 2
        assert summary["env_params"]["max_steps"] == 3

        for key in (
            "return_mean",
            "success_rate",
            "animal_lost_rate",
            "truncated_rate",
            "battery_depleted_rate",
            "battery_spent_mean",
            "steps_to_success_mean",
            "effective_barriers_installed_mean",
        ):
            assert key in summary
    finally:
        env.close()
