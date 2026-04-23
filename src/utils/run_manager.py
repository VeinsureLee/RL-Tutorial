"""
RunContext：把一次训练/测试需要的所有产物路径打包成一个对象。

调用方只需要：
    run = RunContext.new(algo="madqn", num_agents=4, extra={"omega": 1.0, "lr": 1e-4})

就能得到（已自动 mkdir 好的）：
    run.dir           experiments/runs/<YYYYMMDD>_<HHMM>_<algo>_K<N>_...
    run.figs_dir      <run.dir>/figs        — 训练曲线
    run.test_dir      <run.dir>/test        — 测试产物（nav/signal GIF/PNG）
    run.log_path      <run.dir>/run.log     — 训练日志
    run.config_path   <run.dir>/config.snapshot.yml
    run.metrics_path  <run.dir>/metrics.csv
    run.model_path    <run.dir>/model.pth
    run.summary_path  <run.dir>/summary.md

以及三个辅助方法：
    run.dump_config_snapshot(env_cfg, rl_cfg) — 把两份 dict 序列化成 YAML 快照
    run.write_metrics_csv(history)            — trainer.history dict -> 行式 CSV
    run.write_summary_stub(final_stats)       — 自动生成 summary.md 骨架
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from utils.path_tool import get_abs_path


def _format_extra_key(k: str, v: Any) -> str:
    """把 (key, value) 压成 'keyvalue' 的命名片段，避免特殊字符。"""
    if isinstance(v, float):
        # 1e-4 -> 1e-4；1.0 -> 1.0；避免 0.0001 这种长数
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            val = f"{v:.0e}"
        else:
            val = f"{v:g}"
    elif isinstance(v, int):
        val = str(v)
    else:
        val = str(v).replace(" ", "").replace("/", "-")
    return f"{k}{val}"


@dataclass
class RunContext:
    """一次 run 的目录与产物路径。通常由 RunContext.new(...) 构造。"""
    name: str
    dir: str
    figs_dir: str
    test_dir: str
    log_path: str
    config_path: str
    metrics_path: str
    model_path: str
    summary_path: str
    timestamp: str
    algo: str
    num_agents: int
    extra: Dict[str, Any] = field(default_factory=dict)
    tag: Optional[str] = None

    @classmethod
    def new(
        cls,
        algo: str,
        num_agents: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        tag: Optional[str] = None,
        timestamp: Optional[str] = None,
        root_rel: str = "experiments/runs",
    ) -> "RunContext":
        """
        创建新的 run。name 格式：
            <YYYYMMDD>_<HHMM>_<algo>_K<N>[_<keyvalue>...][_<tag>]
        示例：
            20260425_1030_madqn_K4_omega1.0_lr1e-4
            20260425_1200_madqn_K4_omega0.0_nonoma
        """
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
        parts = [ts, algo, f"K{num_agents}"]
        if extra:
            for k, v in extra.items():
                parts.append(_format_extra_key(k, v))
        if tag:
            parts.append(tag)
        name = "_".join(parts)

        run_dir = get_abs_path(os.path.join(root_rel, name))
        figs_dir = os.path.join(run_dir, "figs")
        test_dir = os.path.join(run_dir, "test")
        for d in (run_dir, figs_dir, test_dir):
            os.makedirs(d, exist_ok=True)

        return cls(
            name=name,
            dir=run_dir,
            figs_dir=figs_dir,
            test_dir=test_dir,
            log_path=os.path.join(run_dir, "run.log"),
            config_path=os.path.join(run_dir, "config.snapshot.yml"),
            metrics_path=os.path.join(run_dir, "metrics.csv"),
            model_path=os.path.join(run_dir, "model.pth"),
            summary_path=os.path.join(run_dir, "summary.md"),
            timestamp=ts,
            algo=algo,
            num_agents=num_agents,
            extra=dict(extra or {}),
            tag=tag,
        )

    # ------------------------------------------------------------- 产物落盘
    def dump_config_snapshot(
        self,
        env_cfg: Mapping[str, Any],
        rl_cfg: Mapping[str, Any],
    ) -> str:
        """
        把 env / rl 两份配置序列化成 YAML 快照。
        复杂类型（numpy 数组、tuple、list of tuple）降级为 Python 原生结构。
        """
        import yaml
        import numpy as np

        def _coerce(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, tuple):
                return [_coerce(x) for x in v]
            if isinstance(v, list):
                return [_coerce(x) for x in v]
            if isinstance(v, dict):
                return {k: _coerce(val) for k, val in v.items()}
            return v

        data = {
            "run_name": self.name,
            "timestamp": self.timestamp,
            "algo": self.algo,
            "num_agents": self.num_agents,
            "extra": _coerce(self.extra),
            "tag": self.tag,
            "env_cfg": _coerce(dict(env_cfg)),
            "rl_cfg": _coerce(dict(rl_cfg)),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return self.config_path

    def write_metrics_csv(self, history: Mapping[str, Any]) -> str:
        """
        把 trainer.train() 返回的 history dict 写成行式 CSV。
        列：episode, return, step_return, approach_return, comm_return, ber
        每 agent 的分列: agent{i}_return / agent{i}_ber / ...
        """
        ret = list(history.get("return_list", []))
        step = list(history.get("step_return_list", []))
        approach = list(history.get("approach_return_list", []))
        comm = list(history.get("comm_return_list", []))
        ber = list(history.get("ber_list", []))

        agent_ret = history.get("agent_return_lists", [])
        agent_step = history.get("agent_step_return_lists", [])
        agent_approach = history.get("agent_approach_return_lists", [])
        agent_comm = history.get("agent_comm_return_lists", [])
        agent_ber = history.get("agent_ber_lists", [])
        num_agents = len(agent_ret)

        n = len(ret)
        header = ["episode", "return", "step_return", "approach_return", "comm_return", "neg_log_ber_mean"]
        for i in range(num_agents):
            header.extend([
                f"agent{i}_return",
                f"agent{i}_step",
                f"agent{i}_approach",
                f"agent{i}_comm",
                f"agent{i}_neg_log_ber",
            ])

        def _safe(lst, idx, default=0.0):
            return lst[idx] if idx < len(lst) else default

        with open(self.metrics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ep in range(n):
                row = [
                    ep + 1,
                    _safe(ret, ep),
                    _safe(step, ep),
                    _safe(approach, ep),
                    _safe(comm, ep),
                    _safe(ber, ep),
                ]
                for i in range(num_agents):
                    row.extend([
                        _safe(agent_ret[i], ep) if i < len(agent_ret) else 0.0,
                        _safe(agent_step[i], ep) if i < len(agent_step) else 0.0,
                        _safe(agent_approach[i], ep) if i < len(agent_approach) else 0.0,
                        _safe(agent_comm[i], ep) if i < len(agent_comm) else 0.0,
                        _safe(agent_ber[i], ep) if i < len(agent_ber) else 0.0,
                    ])
                writer.writerow(row)
        return self.metrics_path

    def write_summary_stub(
        self,
        *,
        final_stats: Optional[Mapping[str, Any]] = None,
        notes: str = "",
    ) -> str:
        """
        生成 summary.md 骨架；final_stats 填到"关键数字"区；notes 留给用户手填意图与结论。
        """
        stats = dict(final_stats or {})
        lines = [
            f"# {self.name}",
            "",
            f"- **算法**: `{self.algo}`",
            f"- **K (agents)**: {self.num_agents}",
            f"- **时间戳**: {self.timestamp}",
        ]
        if self.extra:
            extra_str = ", ".join(f"`{k}={v}`" for k, v in self.extra.items())
            lines.append(f"- **关键超参**: {extra_str}")
        if self.tag:
            lines.append(f"- **tag**: `{self.tag}`")
        lines.extend(["", "## 本次实验意图", "",
                      "> TODO: 一句话写清楚——本次 run 想验证什么？相对 baseline 改了什么？", ""])
        lines.append("## 关键数字")
        lines.append("")
        if stats:
            for k, v in stats.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:.4f}")
                else:
                    lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- （trainer 未提供 final_stats）")
        lines.extend(["", "## 结论", "",
                      "> TODO: 一句话写清楚——是否达到预期？下一步该怎么跑？", ""])
        lines.append("## 产物")
        lines.append("")
        lines.extend([
            f"- [run.log](run.log)",
            f"- [config.snapshot.yml](config.snapshot.yml)",
            f"- [metrics.csv](metrics.csv)",
            f"- [model.pth](model.pth)",
            f"- [figs/](figs/)",
            f"- [test/](test/)",
        ])
        if notes:
            lines.extend(["", "## 备注", "", notes, ""])

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return self.summary_path


__all__ = ["RunContext"]
