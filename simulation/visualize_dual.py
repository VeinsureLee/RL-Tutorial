"""
双视图可视化：左图为路径损耗默认热力图，叠加起点、终点、障碍物与轨迹；
右图为普通地图，显示起点（圆点）、终点（五角星）、轨迹，下方用颜色标出对应 agent。支持 GIF 与最后一帧保存。
"""
import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.yml_config import get_channel_config, get_map_and_scenario

_param_parser = get_channel_config()
_config_map_size, _, _, _, _ = get_map_and_scenario()
config_map_size = tuple(int(x) for x in _config_map_size)
param_parser = _param_parser

# 各 agent 使用的颜色与风格（与右图一致）
COLORS = ["blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]


def _get_antenna_pos_for_env(env):
    """将 map_config 下的天线坐标按 env 网格尺寸缩放。"""
    ant = _param_parser.parse_args().antenna_position
    config_rows, config_cols = config_map_size[0], config_map_size[1]
    r_scale = env.grid_rows / max(1, config_rows)
    c_scale = env.grid_cols / max(1, config_cols)
    return (int(round(ant[0] * r_scale)), int(round(ant[1] * c_scale)))


def get_path_loss_map(env):
    """使用 RadioMap 逻辑构建与 env 同尺寸的路径损耗图 (dB)。"""
    from radio_map.radio_map import RadioMap
    antenna_pos = _get_antenna_pos_for_env(env)
    radio = RadioMap(
        map_size_=(env.grid_rows, env.grid_cols),
        grid_size_m=env.grid_size,
        antenna_pos=antenna_pos,
    )
    return radio.build_path_loss_map(los_nlos_getter=env.get_los_nlos)


def get_path_loss_map_for_display(env, path_loss_map=None):
    """路径损耗取相反数用于显示（viridis：高=亮、低=暗）；障碍物设为最暗。"""
    if path_loss_map is None:
        path_loss_map = get_path_loss_map(env)
    pl = np.asarray(path_loss_map, dtype=np.float64).copy()
    display = -pl
    valid = display[np.isfinite(display)]
    vmin = np.min(valid) if len(valid) > 0 else -100.0
    for x, y in env.forbidden_states:
        if 0 <= x < env.grid_rows and 0 <= y < env.grid_cols:
            display[x, y] = vmin - 20
    return display


def _draw_overlay_like_right(env, ax, traj_up_to_frame=None, colors=None):
    """
    在 ax 上按右图风格绘制：起点（圆点）、终点（五角星）、轨迹、当前点（方块）。
    用于左图热力图上的叠加与右图。traj_up_to_frame 为 None 表示完整轨迹。
    """
    if colors is None:
        colors = COLORS
    rows, cols = env.grid_rows, env.grid_cols
    for i, (start_state, target_state) in enumerate(zip(env.start_states, env.target_states)):
        color = colors[i % len(colors)]
        if isinstance(start_state, (list, tuple)) and len(start_state) == 2:
            sx, sy = start_state[0], start_state[1]
            ax.scatter([sy], [sx], c=color, marker="o", s=100, edgecolors="black", linewidths=1, zorder=5)
        if isinstance(target_state, (list, tuple)) and len(target_state) == 2:
            tx, ty = target_state[0], target_state[1]
            ax.scatter([ty], [tx], c=color, marker="*", s=180, edgecolors="black", linewidths=1, zorder=5)
    for i, agent_traj in enumerate(env.traj):
        if len(agent_traj) == 0:
            continue
        if traj_up_to_frame is not None:
            end_idx = min(traj_up_to_frame + 1, len(agent_traj))
            traj_show = agent_traj[:end_idx]
        else:
            traj_show = agent_traj
        if len(traj_show) == 0:
            continue
        color = colors[i % len(colors)]
        arr = np.array(traj_show)
        ax.plot(arr[:, 1], arr[:, 0], "-", color=color, linewidth=2, alpha=0.6, zorder=3)
        last = traj_show[-1]
        ax.scatter(
            [last[1]],
            [last[0]],
            c=color,
            marker="s",
            s=80,
            edgecolors="black",
            linewidths=1,
            zorder=6,
        )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)


def _draw_map_with_start_target(env, ax, traj_up_to_frame=None, colors=None):
    """
    右图：地图（禁区灰）+ 起点（圆点）、终点（五角星）、轨迹、当前点；风格与左图叠加一致。
    """
    if colors is None:
        colors = COLORS
    map_array = np.ones((env.grid_rows, env.grid_cols))
    for x, y in env.forbidden_states:
        if 0 <= x < env.grid_rows and 0 <= y < env.grid_cols:
            map_array[x, y] = 0
    ax.imshow(
        map_array,
        cmap="gray",
        origin="lower",
        extent=[0, env.grid_cols, 0, env.grid_rows],
        interpolation="nearest",
    )
    _draw_overlay_like_right(env, ax, traj_up_to_frame=traj_up_to_frame, colors=colors)


def render_dual(env, path_loss_map=None, save_path=None):
    """
    左图：路径损耗默认热力图，叠加起点、终点、障碍物与轨迹。
    右图：普通地图，显示起点（圆点）、终点（五角星）、轨迹；下方用颜色标出对应 agent。
    """
    if path_loss_map is None:
        path_loss_map = get_path_loss_map(env)
    pl_display = get_path_loss_map_for_display(env, path_loss_map)
    rows, cols = pl_display.shape
    valid = pl_display[np.isfinite(pl_display) & (pl_display > -1e9)]
    vmin = float(np.percentile(valid, 1)) if len(valid) > 0 else -100.0
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else -40.0
    vmin, vmax = min(-80, vmin), max(0, vmax)

    fig, (ax_heat, ax_map) = plt.subplots(1, 2, figsize=(14, 6))

    # 左：路径损耗取相反数，viridis（高=亮、低=暗）
    im = ax_heat.imshow(
        pl_display,
        cmap="turbo",
        origin="lower",
        extent=[0, cols, 0, rows],
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax_heat, label="路径损耗 (dB)")
    ax_heat.set_xlabel("x(单位栅格)")
    ax_heat.set_ylabel("y(单位栅格)")
    ax_heat.set_title("路径损耗")
    _draw_overlay_like_right(env, ax_heat, traj_up_to_frame=None)

    # 右：普通图，显示起点（圆点）、终点（五角星）、轨迹
    ax_map.set_xlabel("y(单位栅格)")
    ax_map.set_ylabel("x(单位栅格)")
    ax_map.set_title("轨迹")
    _draw_map_with_start_target(env, ax_map, traj_up_to_frame=None)

    # 下方图例：按颜色标出对应 agent
    legend_elements = [
        Patch(facecolor=COLORS[i % len(COLORS)], edgecolor="black", label=f"Agent {i + 1}")
        for i in range(env.num_agents)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(env.num_agents, 8),
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def render_animation_dual(
    env,
    path_loss_map=None,
    interval=200,
    save_gif_path=None,
    save_last_frame_path=None,
    max_frames=100,
):
    """
    双视图动画：左图为路径损耗默认热力图，叠加起点、终点、障碍物与随帧更新的移动过程；
    右图显示起点（圆点）、终点（五角星）、轨迹；下方用颜色标出对应 agent。支持保存 GIF 和最后一帧。
    """
    if len(env.traj) == 0:
        print("警告: 没有轨迹数据，请先运行 step。")
        return None

    if path_loss_map is None:
        path_loss_map = get_path_loss_map(env)
    pl_display = get_path_loss_map_for_display(env, path_loss_map)
    rows, cols = pl_display.shape
    valid = pl_display[np.isfinite(pl_display) & (pl_display > -1e9)]
    vmin = float(np.percentile(valid, 1)) if len(valid) > 0 else -100.0
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else -40.0
    vmin, vmax = min(-80, vmin), max(0, vmax)
    max_traj_length = max(len(t) for t in env.traj) if env.traj else 0
    colors = COLORS

    fig, (ax_heat, ax_map) = plt.subplots(1, 2, figsize=(14, 6))

    # 左：路径损耗取相反数，viridis（高=亮、低=暗）
    im = ax_heat.imshow(
        pl_display,
        cmap="turbo",
        origin="lower",
        extent=[0, cols, 0, rows],
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax_heat, label="路径损耗 (dB)")
    ax_heat.set_xlabel("x(单位栅格)")
    ax_heat.set_ylabel("y(单位栅格)")
    ax_heat.set_title("路径损耗")
    ax_heat.set_xlim(-0.5, cols - 0.5)
    ax_heat.set_ylim(-0.5, rows - 0.5)
    # 左图静态：起点、终点（每 agent 不同颜色）
    for i, (start_state, target_state) in enumerate(zip(env.start_states, env.target_states)):
        c = colors[i % len(colors)]
        if isinstance(start_state, (list, tuple)) and len(start_state) == 2:
            ax_heat.scatter([start_state[1]], [start_state[0]], c=c, marker="o", s=100, edgecolors="black", linewidths=1, zorder=5)
        if isinstance(target_state, (list, tuple)) and len(target_state) == 2:
            ax_heat.scatter([target_state[1]], [target_state[0]], c=c, marker="*", s=180, edgecolors="black", linewidths=1, zorder=5)
    # 左图动态：轨迹线与当前点
    heat_lines = []
    heat_points = []
    for i in range(env.num_agents):
        c = colors[i % len(colors)]
        line, = ax_heat.plot([], [], "-", color=c, linewidth=2, alpha=0.6, zorder=3)
        point, = ax_heat.plot([], [], "s", color=c, markersize=8, zorder=6)
        heat_lines.append(line)
        heat_points.append(point)

    # 右：地图底图（禁区）+ 起点、终点
    map_array = np.ones((env.grid_rows, env.grid_cols))
    for x, y in env.forbidden_states:
        if 0 <= x < env.grid_rows and 0 <= y < env.grid_cols:
            map_array[x, y] = 0
    ax_map.imshow(
        map_array,
        cmap="gray",
        origin="lower",
        extent=[0, env.grid_cols, 0, env.grid_rows],
        interpolation="nearest",
    )
    ax_map.set_xlabel("y(单位栅格)")
    ax_map.set_ylabel("x(单位栅格)")
    ax_map.set_title("轨迹")
    ax_map.set_xlim(-0.5, env.grid_cols - 0.5)
    ax_map.set_ylim(-0.5, env.grid_rows - 0.5)
    for i, (start_state, target_state) in enumerate(zip(env.start_states, env.target_states)):
        c = colors[i % len(colors)]
        if isinstance(start_state, (list, tuple)) and len(start_state) == 2:
            ax_map.scatter([start_state[1]], [start_state[0]], c=c, marker="o", s=100, edgecolors="black", linewidths=1, zorder=5)
        if isinstance(target_state, (list, tuple)) and len(target_state) == 2:
            ax_map.scatter([target_state[1]], [target_state[0]], c=c, marker="*", s=180, edgecolors="black", linewidths=1, zorder=5)
    # 右图动态：轨迹线与当前点
    lines = []
    points = []
    for i in range(env.num_agents):
        c = colors[i % len(colors)]
        line, = ax_map.plot([], [], "-", color=c, linewidth=2, alpha=0.6, zorder=3)
        point, = ax_map.plot([], [], "s", color=c, markersize=8, zorder=6)
        lines.append(line)
        points.append(point)

    title_text = ax_map.set_title("")

    def animate(frame):
        for i, agent_traj in enumerate(env.traj):
            if frame < len(agent_traj):
                traj_so_far = agent_traj[: frame + 1]
                arr = np.array(traj_so_far)
                lines[i].set_data(arr[:, 1], arr[:, 0])
                heat_lines[i].set_data(arr[:, 1], arr[:, 0])
                cur = traj_so_far[-1]
                points[i].set_data([cur[1]], [cur[0]])
                heat_points[i].set_data([cur[1]], [cur[0]])
            elif len(agent_traj) > 0:
                arr = np.array(agent_traj)
                lines[i].set_data(arr[:, 1], arr[:, 0])
                heat_lines[i].set_data(arr[:, 1], arr[:, 0])
                last = agent_traj[-1]
                points[i].set_data([last[1]], [last[0]])
                heat_points[i].set_data([last[1]], [last[0]])
        title_text.set_text(f"Step: {frame + 1}/{max_traj_length}")
        return lines + points + heat_lines + heat_points + [title_text]

    # 下方图例：按颜色标出对应 agent
    legend_elements = [
        Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=f"Agent {i + 1}")
        for i in range(env.num_agents)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(env.num_agents, 8),
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    anim = FuncAnimation(
        fig, animate, frames=max_traj_length, interval=interval, blit=True, repeat=True
    )

    if save_gif_path or save_last_frame_path:
        if max_traj_length > max_frames:
            key_frames = [0]
            if max_traj_length > 1:
                step = (max_traj_length - 1) / (max_frames - 1)
                for i in range(1, max_frames - 1):
                    idx = int(round(i * step))
                    if idx < max_traj_length and idx not in key_frames:
                        key_frames.append(idx)
                if max_traj_length - 1 not in key_frames:
                    key_frames.append(max_traj_length - 1)
            frame_indices = key_frames
        else:
            frame_indices = list(range(max_traj_length))

        frames_pil = []
        for frame_idx in tqdm(frame_indices, desc="渲染帧", total=len(frame_indices)):
            animate(frame_idx)
            fig.canvas.draw()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            frames_pil.append(img.copy())
            buf.close()

        if save_last_frame_path and len(frames_pil) > 0:
            last_idx = frame_indices[-1] if frame_indices else 0
            # 用最后一帧的完整轨迹再画一帧，保证“最后一帧”是终点状态
            animate(max_traj_length - 1)
            fig.canvas.draw()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            last_img = Image.open(buf)
            last_img.save(save_last_frame_path)
            buf.close()
            print(f"最后一帧已保存: {save_last_frame_path}")

        if save_gif_path and len(frames_pil) > 0:
            frames_pil[0].save(
                save_gif_path,
                save_all=True,
                append_images=frames_pil[1:],
                duration=interval,
                loop=0,
            )
            print(f"GIF 已保存: {save_gif_path} (共 {len(frames_pil)} 帧)")

    plt.show()
    return anim
