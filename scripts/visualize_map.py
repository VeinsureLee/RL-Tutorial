"""地图可视化工具：把 maps/<name>.yml 渲染成 PNG。

用法::
    python scripts/visualize_map.py                  # 默认地图 -> maps/preview/default.png
    python scripts/visualize_map.py --map default --scale 30
    python scripts/visualize_map.py --map default --out /tmp/map.png

输出说明：
    - 灰色      墙
    - 浅灰      空地
    - 棕黄      门
    - 蓝/红圆点 智能体起点（多个 agent 按颜色区分）
    - 绿色方块  目标点
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.indoor_env import IndoorEnv
from envs.map_builder import DOOR, EMPTY, WALL, build_grid_array, load_map_spec
from utils.paths import project_root


def _build_env_for_render(map_name: str) -> IndoorEnv:
    spec = load_map_spec(map_name)
    cfg = {
        "env": {
            "map_file": map_name,
            "observation_mode": "full",
            "partial_view_size": 7,
            "reward_mode": "independent",
            "reward_goal": 10.0,
            "reward_step": -0.01,
            "reward_collision": -1.0,
            "reward_team_bonus": 5.0,
        },
        "seed": 0,
    }
    env = IndoorEnv(cfg)
    env.reset()
    return env


def _make_image(env: IndoorEnv, scale: int, show_labels: bool) -> Image.Image:
    rgb = env.render()
    img = Image.fromarray(rgb, mode="RGB")
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    if not show_labels:
        return img

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(10, scale // 2))
    except OSError:
        font = ImageFont.load_default()

    # 标注智能体
    for i, (r, c) in enumerate(env.agent_positions):
        x = c * scale + scale // 2
        y = r * scale + scale // 2
        draw.text((x - scale // 4, y - scale // 4), f"A{i}", fill="white", font=font)
    # 标注目标
    for i, (r, c) in enumerate(env.spec["goals"]):
        x = c * scale + scale // 2
        y = r * scale + scale // 2
        draw.text((x - scale // 4, y - scale // 4), f"G{i}", fill="black", font=font)
    return img


def _summary_strings(env: IndoorEnv) -> list[str]:
    spec = env.spec
    return [
        f"Map size      : {spec['size'][0]} x {spec['size'][1]}",
        f"Rooms         : {len(spec['rooms'])}",
        f"Doors         : {len(spec['doors'])}",
        f"Agents        : {spec['num_agents']}  starts={spec['agents_start']}",
        f"Goals         : {spec['goals']}",
        f"Walls / cells : {(env.grid == WALL).sum()} / {env.grid.size}",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="default", help="Map file name (without .yml)")
    p.add_argument("--scale", type=int, default=30, help="Pixel scale per grid cell")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--no-labels", action="store_true", help="Disable A0/G0 text labels")
    args = p.parse_args()

    env = _build_env_for_render(args.map)
    img = _make_image(env, args.scale, show_labels=not args.no_labels)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = project_root() / "maps" / "preview" / f"{args.map}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    print(f"Saved: {out_path}")
    for line in _summary_strings(env):
        print(f"  {line}")


if __name__ == "__main__":
    main()
