"""Smoke tests for the Stage 3 random baseline script."""

from experiments.random_baseline import evaluate_random_policy


def test_evaluate_random_policy_returns_stage3_metrics():
    episodes, summary = evaluate_random_policy(
        episodes=5,
        seed=123,
        env_kwargs={
            "max_steps": 5,
            "flood_advance_prob": 0.0,
            "flood_deepen_prob": 0.0,
        },
    )

    assert len(episodes) == 5
    assert summary["episodes"] == 5
    assert summary["seed"] == 123
    assert summary["env_params"]["max_steps"] == 5

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
