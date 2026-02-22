"""
Radio Map 实现：基于 communication.channel 的 path loss 计算，并以热力图展示。
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from communication.channel import path_loss
from config.yml_config import get_map_and_scenario, _get_env_parser


def _get_grid_size_m():
    """获取网格物理边长（米），默认 0.4。"""
    try:
        return float(_get_env_parser().parse_args().grid_size)
    except Exception:
        return 0.4


def _get_map_and_los_nlos():
    """延迟加载地图与 LOS/NLOS 函数。"""
    map_size, forbidden_areas, get_los_nlos, antenna_position, _ = get_map_and_scenario()
    return map_size, get_los_nlos, antenna_position


class RadioMap:
    """
    基于 communication.channel.path_loss 的 Radio Map。
    地图以离散栅格表示，每个格点使用 LOS/NLOS 模型计算路径损耗 (dB)，并支持热力图展示。
    """

    def __init__(self,
                 map_size_=None,
                 grid_size_m=None,
                 antenna_pos=None):
        """
        :param map_size_: 地图尺寸 (rows, cols)，离散格点数，默认使用 config.map_config.map_size
        :param grid_size_m: 每个格点对应的物理边长（米），默认使用 env_arguments.grid_size（如 0.4）
        :param antenna_pos: 天线/基站在地图上的离散坐标 (x, y)，默认使用 map_config.antenna_position
        """
        _ms, _get_ln, _ap = _get_map_and_los_nlos()
        self.map_size = tuple(map_size_) if map_size_ is not None else tuple(_ms)
        self.grid_size_m = grid_size_m if grid_size_m is not None else _get_grid_size_m()
        if antenna_pos is not None:
            self.antenna_pos = (int(antenna_pos[0]), int(antenna_pos[1]))
        else:
            self.antenna_pos = (int(_ap[0]), int(_ap[1]))
        self.path_loss_map = None

    def build_path_loss_map(self, los_nlos_getter=None):
        """
        使用 communication.channel.path_loss 构建整张地图的路径损耗 (dB)。
        :param los_nlos_getter: 可选，函数 (i, j) -> 'los'|'nlos'，默认使用 config.map_config.get_los_nlos
        :return: 2D 数组，形状为 map_size，单位为 dB
        """
        rows, cols = self.map_size[0], self.map_size[1]
        _, get_los_nlos, _ = _get_map_and_los_nlos()
        get_region = los_nlos_getter if los_nlos_getter is not None else get_los_nlos
        pl_map = np.zeros((rows, cols), dtype=np.float64)

        for i in range(rows):
            for j in range(cols):
                region_type = get_region(i, j)
                pl_map[i, j] = path_loss(
                    (i, j),
                    region_type,
                    grid_size_m=self.grid_size_m,
                    antenna_pos=self.antenna_pos,
                )
        self.path_loss_map = pl_map
        return pl_map

    def show_heatmap(self, path_loss_map=None, title="Path Loss (dB)", figsize=(8, 6), save_path=None):
        """
        将路径损耗地图以热力图展示。
        :param path_loss_map: 2D 数组 (dB)，若为 None 则使用 self.path_loss_map
        :param title: 图标题
        :param figsize: 图像尺寸
        :param save_path: 可选，保存路径（如 'path_loss_heatmap.png'）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("热力图展示需要 matplotlib，请安装: pip install matplotlib")

        data = path_loss_map if path_loss_map is not None else self.path_loss_map
        if data is None:
            data = self.build_path_loss_map()

        rows, cols = data.shape
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            data,
            cmap="viridis",
            origin="upper",
            extent=[0, cols, rows, 0],
            aspect="equal",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, label="Path Loss (dB)")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.set_title(title)
        ant_x, ant_y = self.antenna_pos[0], self.antenna_pos[1]
        ax.plot(ant_y + 0.5, ant_x + 0.5, "r*", markersize=12, label="Antenna")
        ax.legend(loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig, ax


if __name__ == "__main__":
    radio_map = RadioMap()
    pl_map = radio_map.build_path_loss_map()
    print("Path loss map shape:", pl_map.shape)
    print("Path loss range (dB):", pl_map.min(), "~", pl_map.max())
    radio_map.show_heatmap(title="Path Loss Map (dB)", save_path="path_loss_heatmap.png")
