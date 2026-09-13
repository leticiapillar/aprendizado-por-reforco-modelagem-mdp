"""Testes da renderizacao do FloodGuardEnv (Etapa 2)."""

import numpy as np
import pytest

from floodguard.envs.flood_guard_env import FloodGuardEnv
from floodguard.rendering.renderer import render_frame


def test_render_frame_returns_rgb_array():
    env = FloodGuardEnv()
    env.reset(seed=0)

    frame = render_frame(env)

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_env_render_without_render_mode_warns_and_returns_none():
    env = FloodGuardEnv()  # render_mode=None por padrao
    env.reset(seed=0)

    assert env.render() is None


def test_env_render_rgb_array_matches_renderer():
    env = FloodGuardEnv(render_mode="rgb_array")
    env.reset(seed=0)

    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3


def test_render_reflects_state_changes_across_an_episode():
    env = FloodGuardEnv(render_mode="rgb_array")
    env.reset(seed=0)

    frame_initial = env.render()
    for _ in range(5):
        env.step(env.action_space.sample())
    frame_later = env.render()

    assert not np.array_equal(frame_initial, frame_later)


def test_render_does_not_crash_for_lost_and_saved_animal_states():
    # Estado "perdido"
    env = FloodGuardEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.animal_status = FloodGuardEnv.LOST
    frame_lost = env.render()
    assert frame_lost.shape[2] == 3

    # Estado "com o robo" (badge) e "salvo"
    env.animal_status = FloodGuardEnv.WITH_ROBOT
    frame_carrying = env.render()
    assert frame_carrying.shape[2] == 3

    env.animal_status = FloodGuardEnv.SAVED
    frame_saved = env.render()
    assert frame_saved.shape[2] == 3


@pytest.mark.parametrize("facing", [(0, 1), (0, -1), (-1, 0), (1, 0)])
def test_render_all_facing_directions(facing):
    env = FloodGuardEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env._facing = facing
    frame = env.render()
    assert frame.shape[2] == 3
