"""Gera imagens e um GIF de exemplo do FloodGuardEnv (Etapa 2).

Roda um episodio com uma politica heuristica simples — so para produzir uma
sequencia legivel que passa pelas principais mecanicas do MDP (navegacao,
instalacao de barreira, resgate e entrega) — e salva em `results/figures/`:

  - example_episode_initial.png   (estado inicial)
  - example_episode_barrier.png   (barreira recem-instalada)
  - example_episode_rescue.png    (animal acabou de ser resgatado)
  - example_episode_final.png     (estado final do episodio)
  - example_episode.gif           (episodio completo animado)

Uso: `python experiments/render_example_episode.py` (com o venv ativo).
"""

from pathlib import Path

import numpy as np
from PIL import Image

from floodguard.envs.flood_guard_env import FloodGuardEnv
from floodguard.utils import set_global_seed

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def _step_towards(env: FloodGuardEnv, target: tuple[int, int]) -> int:
    """Acao de movimento que reduz a distancia ate `target` (linha, depois
    coluna), desviando para o outro eixo se a direcao preferida estiver
    bloqueada por agua profunda (script ilustrativo: usa o estado do
    ambiente diretamente, sem tentar navegar com observabilidade parcial).
    """
    r, c = env.robot_pos.tolist()
    tr, tc = target
    candidates = []
    if r != tr:
        dr = -1 if tr < r else 1
        action = FloodGuardEnv.UP if dr == -1 else FloodGuardEnv.DOWN
        candidates.append((action, (r + dr, c)))
    if c != tc:
        dc = -1 if tc < c else 1
        action = FloodGuardEnv.LEFT if dc == -1 else FloodGuardEnv.RIGHT
        candidates.append((action, (r, c + dc)))
    for action, (nr, nc) in candidates:
        if env.flood[nr, nc] != FloodGuardEnv.DEEP:
            return action
    return FloodGuardEnv.WAIT_CHARGE  # ambos os eixos bloqueados: espera (raro)


def _scripted_policy(env: FloodGuardEnv) -> int:
    target = env.animal_start if env.animal_status == FloodGuardEnv.AWAITING else env.safe_zone
    if tuple(env.robot_pos.tolist()) != target:
        return _step_towards(env, target)
    return FloodGuardEnv.WAIT_CHARGE


def run_example_episode(seed: int = 7, max_steps: int = 100):
    set_global_seed(seed)
    env = FloodGuardEnv(max_steps=max_steps, render_mode="rgb_array")
    env.reset(seed=seed)

    frames = [env.render()]
    tagged: dict[str, int] = {}
    info: dict = {}

    # Desvia para uma celula seca adjacente ao foco da enchente e instala uma
    # barreira ali, so para ilustrar o mecanismo na sequencia de imagens. Usa
    # a celula abaixo do foco (nao a linha/coluna do animal ou da base) para
    # que o caminho ate ela nao passe, por coincidencia, pela celula do
    # animal ou da zona segura e dispare um resgate/entrega prematuros.
    barrier_cell = (env.flood_source[0] + 1, env.flood_source[1])
    terminated = truncated = False
    while tuple(env.robot_pos.tolist()) != barrier_cell:
        action = _step_towards(env, barrier_cell)
        _obs, _reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    else:
        _obs, _reward, terminated, truncated, info = env.step(FloodGuardEnv.INSTALL_BARRIER)
        frames.append(env.render())
        tagged["barrier"] = len(frames) - 1

    while not terminated and not truncated:
        action = _scripted_policy(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if env.animal_status == FloodGuardEnv.WITH_ROBOT and "rescue" not in tagged:
            tagged["rescue"] = len(frames) - 1

    tagged["final"] = len(frames) - 1
    return frames, tagged, info


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 4) -> None:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frames, tagged, info = run_example_episode()

    save_frame(frames[0], FIGURES_DIR / "example_episode_initial.png")
    if "barrier" in tagged:
        save_frame(frames[tagged["barrier"]], FIGURES_DIR / "example_episode_barrier.png")
    if "rescue" in tagged:
        save_frame(frames[tagged["rescue"]], FIGURES_DIR / "example_episode_rescue.png")
    save_frame(frames[tagged["final"]], FIGURES_DIR / "example_episode_final.png")
    save_gif(frames, FIGURES_DIR / "example_episode.gif")

    print(f"Episódio com {len(frames)} frames salvo em {FIGURES_DIR}")
    print(f"Resultado final: {info}")


if __name__ == "__main__":
    main()
