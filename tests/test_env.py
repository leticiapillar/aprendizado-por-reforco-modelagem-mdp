"""Testes do MDP do FloodGuardEnv (Etapa 1).

Cobrem: reset determinístico, custos de bateria por movimento, bloqueio de
movimento por água profunda, barreiras bloqueando o avanço da água,
resgate/entrega do animal, perda do animal, bateria zerada fora da base,
truncamento por limite de steps e conformidade com a API do Gymnasium.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import floodguard  # noqa: F401  (garante o registro de "FloodGuard-v0")
import gymnasium as gym
from floodguard.envs.flood_guard_env import FloodGuardEnv


def make_env(**kwargs) -> FloodGuardEnv:
    return FloodGuardEnv(**kwargs)


# ----------------------------------------------------------------------
# Conformidade com a API do Gymnasium
# ----------------------------------------------------------------------


def test_check_env_passes():
    env = make_env()
    check_env(env.unwrapped if hasattr(env, "unwrapped") else env)


def test_registered_env_makeable():
    env = gym.make("FloodGuard-v0")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_random_rollout_never_crashes():
    env = make_env()
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        assert env.observation_space.contains(obs)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            if terminated or truncated:
                break


# ----------------------------------------------------------------------
# reset()
# ----------------------------------------------------------------------


def test_reset_initial_state():
    env = make_env()
    obs, info = env.reset(seed=0)

    assert tuple(obs["robot_pos"]) == env.robot_base
    assert obs["battery"][0] == env.max_battery
    assert obs["barrier_kits"][0] == env.initial_barrier_kits
    assert obs["animal_status"] == FloodGuardEnv.AWAITING
    assert tuple(obs["animal_pos"]) == env.animal_start
    assert obs["steps_left"][0] == env.max_steps

    flood = obs["flood_map"]
    assert flood[env.flood_source] == FloodGuardEnv.SHALLOW
    assert np.sum(flood == FloodGuardEnv.SHALLOW) == 1
    assert np.sum(flood == FloodGuardEnv.DEEP) == 0


def test_reset_is_deterministic_given_seed():
    env_a = make_env()
    env_b = make_env()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)

    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])

    # Mesma seed + mesmas acoes => mesma trajetoria (dinamica estocastica
    # da agua deve ser reprodutivel via self.np_random).
    actions = [FloodGuardEnv.UP, FloodGuardEnv.WAIT_CHARGE, FloodGuardEnv.RIGHT] * 10
    for action in actions:
        obs_a, r_a, t_a, tr_a, _ = env_a.step(action)
        obs_b, r_b, t_b, tr_b, _ = env_b.step(action)
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])
        assert r_a == r_b
        if t_a or tr_a:
            break


# ----------------------------------------------------------------------
# Movimento e bateria
# ----------------------------------------------------------------------


def test_move_into_dry_cell_costs_battery_and_updates_position():
    env = make_env(move_cost_dry=1)
    env.reset(seed=0)
    start_battery = env.battery

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)

    assert tuple(obs["robot_pos"]) == (env.robot_base[0], env.robot_base[1] - 1)
    assert env.battery == start_battery - env.move_cost_dry
    assert reward == pytest.approx(env.step_penalty)
    assert not terminated and not truncated


def test_move_into_shallow_water_costs_more_battery():
    env = make_env()
    env.reset(seed=0)
    shallow_cell = (env.robot_base[0], env.robot_base[1] - 1)
    env.flood[shallow_cell] = FloodGuardEnv.SHALLOW
    start_battery = env.battery

    env.step(FloodGuardEnv.LEFT)

    assert env.battery == start_battery - env.move_cost_shallow


def test_move_blocked_by_deep_water():
    env = make_env()
    env.reset(seed=0)
    deep_cell = (env.robot_base[0], env.robot_base[1] - 1)
    env.flood[deep_cell] = FloodGuardEnv.DEEP
    start_battery = env.battery
    start_pos = tuple(env.robot_pos.tolist())

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)

    assert tuple(obs["robot_pos"]) == start_pos
    assert env.battery == start_battery
    assert not terminated


def test_move_blocked_by_grid_boundary():
    env = make_env(robot_base=(0, 0), animal_start=(5, 5))
    env.reset(seed=0)
    start_battery = env.battery

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.UP)

    assert tuple(obs["robot_pos"]) == (0, 0)
    assert env.battery == start_battery


def test_wait_charge_recovers_battery_only_on_base():
    env = make_env(charge_rate=20)
    env.reset(seed=0)
    env.battery = 50

    env.step(FloodGuardEnv.WAIT_CHARGE)
    assert env.battery == 70

    env.step(FloodGuardEnv.LEFT)  # sai da base
    battery_after_move = env.battery
    env.step(FloodGuardEnv.WAIT_CHARGE)  # fora da base: nao recarrega
    assert env.battery == battery_after_move


def test_battery_is_clamped_to_max():
    env = make_env(max_battery=100, charge_rate=20)
    env.reset(seed=0)
    env.battery = 95

    env.step(FloodGuardEnv.WAIT_CHARGE)

    assert env.battery == 100


def test_info_tracks_stage3_evaluation_metrics():
    env = make_env(
        robot_base=(5, 5),
        animal_start=(5, 4),
        safe_zone=(5, 3),
        flood_advance_prob=0.0,
        flood_deepen_prob=0.0,
    )
    _obs, info = env.reset(seed=0)

    assert info["battery_spent"] == 0
    assert info["barriers_installed"] == 0
    assert info["effective_barriers_installed"] == 0
    assert info["pickup_step"] is None
    assert info["delivery_step"] is None
    assert info["termination_reason"] == "running"

    _obs, _reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)
    assert not terminated and not truncated
    assert info["pickup_step"] == 1
    assert info["battery_spent"] == env.move_cost_dry

    _obs, _reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)
    assert terminated and not truncated
    assert info["is_success"] is True
    assert info["delivery_step"] == 2
    assert info["termination_reason"] == "success"
    assert info["battery_spent"] == 2 * env.move_cost_dry


# ----------------------------------------------------------------------
# Barreiras
# ----------------------------------------------------------------------


def test_install_barrier_consumes_kit_and_battery_and_marks_cell():
    env = make_env(barrier_install_cost=15)
    env.reset(seed=0)
    env.robot_pos = np.array(env.flood_source, dtype=np.int32)  # celula rasa (foco)
    start_battery = env.battery
    start_kits = env.kits_remaining

    obs, reward, *_ = env.step(FloodGuardEnv.INSTALL_BARRIER)

    assert env.barriers[env.flood_source] == 1
    assert env.kits_remaining == start_kits - 1
    assert env.battery == start_battery - env.barrier_install_cost
    assert reward == pytest.approx(env.step_penalty + env.barrier_effective_reward)


def test_install_barrier_noop_without_kits():
    env = make_env(initial_barrier_kits=0)
    env.reset(seed=0)
    start_battery = env.battery

    env.step(FloodGuardEnv.INSTALL_BARRIER)

    assert env.barriers.sum() == 0
    assert env.battery == start_battery


def test_barrier_protects_cell_from_flood_advance():
    env = make_env(flood_advance_prob=1.0, flood_deepen_prob=1.0, initial_barrier_kits=1)
    env.reset(seed=0)
    # Coloca o robo numa celula seca adjacente ao foco e instala barreira nela.
    protected_cell = (env.flood_source[0], env.flood_source[1] + 1)
    env.robot_pos = np.array(protected_cell, dtype=np.int32)
    env.step(FloodGuardEnv.INSTALL_BARRIER)
    assert env.barriers[protected_cell] == 1

    env.step(FloodGuardEnv.WAIT_CHARGE)  # so avanca a agua (p=q=1.0)

    assert env.flood[protected_cell] == FloodGuardEnv.DRY  # protegida, nao inundou
    assert env.flood[env.flood_source] == FloodGuardEnv.DEEP  # sem barreira, avancou


# ----------------------------------------------------------------------
# Resgate, perda do animal, bateria zerada, truncamento
# ----------------------------------------------------------------------


def test_rescue_and_delivery_gives_reward_and_terminates():
    env = make_env(animal_start=(5, 4), safe_zone=(5, 0), robot_base=(5, 5))
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)
    assert env.animal_status == FloodGuardEnv.WITH_ROBOT
    assert not terminated

    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)

    assert env.animal_status == FloodGuardEnv.SAVED
    assert terminated
    assert reward == pytest.approx(env.step_penalty + env.rescue_reward)
    assert info["is_success"] is True


def test_animal_lost_when_its_cell_becomes_deep_water():
    env = make_env(flood_advance_prob=1.0, flood_deepen_prob=1.0)
    env.reset(seed=0)
    env.flood[env.animal_start] = FloodGuardEnv.SHALLOW

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.WAIT_CHARGE)

    assert env.animal_status == FloodGuardEnv.LOST
    assert terminated
    assert reward == pytest.approx(env.step_penalty + env.animal_lost_penalty)


def test_battery_depleted_outside_base_is_terminal():
    env = make_env(move_cost_dry=1)
    env.reset(seed=0)
    env.battery = 1

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.LEFT)

    assert env.battery == 0
    assert terminated
    assert reward == pytest.approx(env.step_penalty + env.battery_depleted_penalty)


def test_battery_zero_on_base_is_not_terminal():
    env = make_env(charge_rate=0)
    env.reset(seed=0)
    env.battery = 0

    obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.WAIT_CHARGE)

    assert env.battery == 0
    assert not terminated


def test_truncation_at_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)

    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(FloodGuardEnv.WAIT_CHARGE)

    assert truncated
    assert not terminated
