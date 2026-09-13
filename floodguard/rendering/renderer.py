"""Renderizacao do FloodGuardEnv via matplotlib.

Desenha o grid seguindo o estilo do mockup em
`assets/floodguard_mockup_simplificado.png`: celulas coloridas por nivel de
inundacao, robo como triangulo (com indicador de bateria), animal como
circulo, zona segura como pentagono, base como quadrado, barreiras como
contorno marrom em volta da celula protegida, e uma linha tracejada ligando
o animal (enquanto nao entregue) a zona segura.

Usa `matplotlib.figure.Figure` + `FigureCanvasAgg` diretamente (sem
`pyplot`), para nao interferir no backend global do matplotlib quando este
modulo e importado dentro de um notebook (Etapa 7) ou de scripts de
treino/experimentos que tambem plotam graficos.
"""

import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, RegularPolygon

DRY_COLOR = "#f2eec9"
SHALLOW_COLOR = "#a9d6e5"
DEEP_COLOR = "#1f6091"
BARRIER_COLOR = "#6b3a1f"
BASE_COLOR = "#1f9e89"
SAFE_ZONE_COLOR = "#4fd1a5"
ANIMAL_COLOR = "#f4805a"
ROBOT_COLOR = "#12203f"
BATTERY_FILL_COLOR = "#4fd1a5"
GRID_LINE_COLOR = "#bbbbbb"
LOST_MARK_COLOR = "#c0392b"

_FLOOD_COLORS = {0: DRY_COLOR, 1: SHALLOW_COLOR, 2: DEEP_COLOR}


def _cell_center(row: int, col: int, grid_size: int) -> tuple[float, float]:
    """Converte (linha, coluna) da matriz para (x, y) no plot (linha 0 no topo)."""
    return col + 0.5, (grid_size - 1 - row) + 0.5


def _draw_flood_cells(ax, env) -> None:
    grid_size = env.grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            x, y = col, grid_size - 1 - row
            color = _FLOOD_COLORS[int(env.flood[row, col])]
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=color, edgecolor=GRID_LINE_COLOR, linewidth=1)
            )


def _draw_barriers(ax, env) -> None:
    grid_size = env.grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            if env.barriers[row, col]:
                x, y = col, grid_size - 1 - row
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        facecolor="none",
                        edgecolor=BARRIER_COLOR,
                        linewidth=5,
                    )
                )


def _draw_base(ax, env) -> None:
    row, col = env.robot_base
    x, y = col, env.grid_size - 1 - row
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=BASE_COLOR, edgecolor="black", linewidth=1.5))
    ax.text(x + 0.5, y + 0.5, "B", ha="center", va="center", color="white", fontsize=14, fontweight="bold")


def _draw_safe_zone(ax, env) -> None:
    row, col = env.safe_zone
    cx, cy = _cell_center(row, col, env.grid_size)
    ax.add_patch(
        RegularPolygon(
            (cx, cy), numVertices=5, radius=0.32, facecolor=SAFE_ZONE_COLOR, edgecolor="black", linewidth=1.5
        )
    )
    ax.text(cx, cy - 0.03, "S", ha="center", va="center", color="white", fontsize=13, fontweight="bold")


def _draw_animal(ax, env) -> None:
    row, col = env.animal_start
    cx, cy = _cell_center(row, col, env.grid_size)

    if env.animal_status == env.AWAITING:
        ax.add_patch(_circle(cx, cy, 0.28, ANIMAL_COLOR))
        ax.text(cx, cy, "A", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    elif env.animal_status == env.LOST:
        ax.plot(
            [cx - 0.22, cx + 0.22], [cy - 0.22, cy + 0.22], color=LOST_MARK_COLOR, linewidth=3, solid_capstyle="round"
        )
        ax.plot(
            [cx - 0.22, cx + 0.22], [cy + 0.22, cy - 0.22], color=LOST_MARK_COLOR, linewidth=3, solid_capstyle="round"
        )


def _circle(cx: float, cy: float, radius: float, color: str) -> Circle:
    return Circle((cx, cy), radius, facecolor=color, edgecolor="black", linewidth=1.5)


def _draw_target_line(ax, env) -> None:
    if env.animal_status not in (env.AWAITING, env.WITH_ROBOT):
        return
    ar, ac = env.animal_start
    sr, sc = env.safe_zone
    ax_, ay_ = _cell_center(ar, ac, env.grid_size)
    bx_, by_ = _cell_center(sr, sc, env.grid_size)
    ax.plot([ax_, bx_], [ay_, by_], linestyle="--", color=ANIMAL_COLOR, linewidth=1.5, alpha=0.7, zorder=1)


def _triangle_vertices(cx: float, cy: float, facing: tuple[int, int], size: float = 0.32):
    dr, dc = facing
    angle = math.atan2(-dr, dc)
    tip = (size, 0.0)
    back_left = (-size * 0.7, size * 0.6)
    back_right = (-size * 0.7, -size * 0.6)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate_translate(pt):
        x, y = pt
        return (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)

    return [rotate_translate(tip), rotate_translate(back_left), rotate_translate(back_right)]


def _draw_battery_indicator(ax, env, cx: float, cy: float) -> None:
    row = int(env.robot_pos[0])
    bar_width, bar_height = 0.6, 0.12
    bar_y = cy + 0.30 if row > 0 else cy - 0.42
    bar_x = cx - bar_width / 2

    fraction = 0.0 if env.max_battery <= 0 else max(0.0, min(1.0, env.battery / env.max_battery))
    ax.add_patch(Rectangle((bar_x, bar_y), bar_width, bar_height, facecolor="white", edgecolor="black", linewidth=1))
    ax.add_patch(
        Rectangle((bar_x, bar_y), bar_width * fraction, bar_height, facecolor=BATTERY_FILL_COLOR, edgecolor="none")
    )


def _draw_robot(ax, env) -> None:
    row, col = env.robot_pos.tolist()
    cx, cy = _cell_center(row, col, env.grid_size)
    facing = getattr(env, "_facing", (0, 1))

    ax.add_patch(Polygon(_triangle_vertices(cx, cy, facing), closed=True, facecolor=ROBOT_COLOR, edgecolor="black"))
    _draw_battery_indicator(ax, env, cx, cy)

    if env.animal_status == env.WITH_ROBOT:
        badge_cx, badge_cy = cx + 0.32, cy + 0.32
        ax.add_patch(_circle(badge_cx, badge_cy, 0.14, ANIMAL_COLOR))
        ax.text(badge_cx, badge_cy, "A", ha="center", va="center", color="white", fontsize=8, fontweight="bold")


def _legend_handles():
    return [
        Patch(facecolor=DRY_COLOR, edgecolor=GRID_LINE_COLOR, label="Célula seca"),
        Patch(facecolor=SHALLOW_COLOR, edgecolor=GRID_LINE_COLOR, label="Água rasa"),
        Patch(facecolor=DEEP_COLOR, edgecolor=GRID_LINE_COLOR, label="Água profunda"),
        Patch(facecolor="none", edgecolor=BARRIER_COLOR, linewidth=3, label="Barreira instalada"),
        Patch(facecolor=BASE_COLOR, edgecolor="black", label="Base (recarga)"),
        Patch(facecolor=SAFE_ZONE_COLOR, edgecolor="black", label="Zona segura"),
        Patch(facecolor=ANIMAL_COLOR, edgecolor="black", label="Animal a resgatar"),
        Patch(facecolor=ROBOT_COLOR, edgecolor="black", label="Robô"),
    ]


_STATUS_LABELS = {0: "aguardando resgate", 1: "com o robô", 2: "salvo", 3: "perdido"}


def _title_text(env) -> str:
    steps = getattr(env, "_steps", 0)
    line1 = f"Step {steps}/{env.max_steps}  |  Bateria {env.battery}/{env.max_battery}  |  Kits: {env.kits_remaining}"
    line2 = f"Animal: {_STATUS_LABELS[env.animal_status]}"
    return f"{line1}\n{line2}"


def draw_frame(ax, env, show_legend: bool = True, show_title: bool = True) -> None:
    """Desenha o estado atual de `env` no eixo `ax` (reutilizavel em figuras compostas)."""
    grid_size = env.grid_size
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_flood_cells(ax, env)
    _draw_barriers(ax, env)
    _draw_base(ax, env)
    _draw_safe_zone(ax, env)
    _draw_target_line(ax, env)
    _draw_animal(ax, env)
    _draw_robot(ax, env)

    if show_title:
        ax.set_title(_title_text(env), fontsize=8)
    if show_legend:
        ax.legend(
            handles=_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=8,
        )


def render_frame(env, figsize=(5, 6.2), dpi: int = 120) -> np.ndarray:
    """Renderiza o estado atual de `env` e retorna um array RGB (H, W, 3) uint8."""
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.02, 0.16, 0.96, 0.72))
    draw_frame(ax, env)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return buf[:, :, :3].copy()
