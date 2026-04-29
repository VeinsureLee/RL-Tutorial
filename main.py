"""
项目入口：完成参数解析、环境 & 模型构建、模式分派（train / test）。

用法示例::
    python main.py --model madqn --mode train
    python main.py --model dqn   --mode train --num_episodes 200 --lr 5e-5
    python main.py --model madqn --mode test  --model_path experiments/runs/<name>/model.pth
    python main.py --model dqn   --mode test  --max_steps 500
    python main.py --model dqn   --mode train --placement custom   # 交互式起终点选择

参数优先级: 命令行 > config/base/rl.yml > 代码内默认。

产物布局（每次运行 = 一个文件夹）::
    experiments/runs/<YYYYMMDD_HHMM>_<algo>_K<N>_<keyvals>[_<tag>]/
        run.log                 训练 / 测试日志
        config.snapshot.yml     env_cfg + rl_cfg 的 YAML 快照
        metrics.csv             trainer.history 全量
        model.pth               权重
        figs/                   训练曲线（仅 train 模式）
        test/                   nav/signal 的 GIF + PNG（仅 test 模式 / train 末尾可选）
        summary.md              手写意图与结论（自动生成骨架）
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
# 代码已迁移到 src/；把 src/ 放到 sys.path，让 `from config.xxx` / `from env.xxx` 继续生效。
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch

from config.yml_config import get_env_config, get_rl_config

# env / rl_algorithms 的导入延迟到 main() 内部执行，原因：
# env/env.py 和 rl_algorithms/plot.py 在模块加载时调用 matplotlib.use("Agg")，
# 会将 matplotlib 后端锁定为非交互模式；而 --placement custom 需要在此之前
# 弹出交互式图形窗口选择 agent 起终点。延迟导入确保可视化在 Agg 锁定前完成。
# torch 本身不触碰 matplotlib，可以安全地在模块层级导入。


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Robot DRL Navigation (DQN / MADQN / SharedMADQN / QMIX)")
    p.add_argument("--model",
                   choices=["dqn", "madqn", "shared_madqn", "qmix"],
                   default="shared_madqn",
                   help="algorithm: dqn | madqn (independent) | shared_madqn (parameter sharing) | qmix")
    p.add_argument("--mode", choices=["train", "test"], default="train", help="mode")

    # 训练超参（命令行覆盖 rl.yml）
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--epsilon_min", type=float, default=None)
    p.add_argument("--epsilon_decay", type=float, default=None)
    p.add_argument("--num_iterations", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--episode_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--update_freq", type=int, default=None)
    p.add_argument("--replay_buffer_size", type=int, default=None)
    p.add_argument("--train_interval", type=int, default=None,
                   help="gradient update interval (env steps); None -> rl.yml value")

    # 测试参数
    p.add_argument("--model_path", type=str, default=None, help="path to .pth for test")
    p.add_argument("--max_steps", type=int, default=None, help="max steps during test")

    # run 目录命名
    p.add_argument("--tag", type=str, default=None,
                   help="optional tag appended to the run folder name")

    # 通用
    p.add_argument("--agent_id", type=int, default=0, help="DQN only: which agent to train")
    p.add_argument("--no_save_model", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_test_after_train", action="store_true",
                   help="skip the auto test episode at the end of training")
    p.add_argument("--randomize_reset", choices=["auto", "true", "false"], default="false",
                   help="resample (start, target) every env.reset(); "
                        "auto=true for shared_madqn / false otherwise (default), "
                        "true/false to override")
    p.add_argument("--placement", choices=["default", "custom"], default="custom",
                   help="agent placement: default = use scenario.npz fixed states (or random if "
                        "randomize_on_reset=True) | custom = open interactive visual selector "
                        "to click start/target positions for each agent")
    p.add_argument("--comm_model", choices=["noma", "ofdma"], default=None,
                   help="communication access scheme: noma (default) | ofdma")
    return p.parse_args()


def _resolve_cfg(args: argparse.Namespace) -> dict:
    """命令行 > yml 合并：非 None 的命令行字段覆盖 yml 默认值。"""
    overrides = {k: v for k, v in vars(args).items() if v is not None and k in {
        "lr", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
        "num_iterations", "num_episodes", "episode_length",
        "batch_size", "hidden_dim", "update_freq", "replay_buffer_size",
        "train_interval",
    }}
    return get_rl_config(algo=args.model, **overrides)


def _build_model(algo: str, env, cfg: dict, device: torch.device, agent_id: int = 0):
    common = dict(
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        hidden_dim=cfg["hidden_dim"], update_freq=cfg["update_freq"],
        replay_buffer_size=cfg["replay_buffer_size"], device=device,
    )
    if algo == "dqn":
        return DQN(env, agent_id=agent_id, **common)
    if algo == "qmix":
        return QMIX(env, **common)
    if algo == "shared_madqn":
        return SharedMADQN(env, **common)
    return MADQN(env, **common)


def _make_run(args, cfg: dict, env, mode: str):
    """构造 RunContext；extra 字段自动包含奖励权重 ω 和学习率。"""
    extra = {
        "omega": env.omega,
        "lr": cfg["lr"],
    }
    tag = args.tag
    if mode == "test" and not tag:
        tag = "test"
    return RunContext.new(
        algo=args.model, num_agents=env.num_agents,
        extra=extra, tag=tag,
    )


def _run_train(args, cfg: dict, env, device: torch.device, env_cfg: dict) -> dict:
    run = _make_run(args, cfg, env, mode="train")
    print(f"[run] {run.name}")
    print(f"[run] dir = {run.dir}")

    run.dump_config_snapshot(env_cfg, cfg)

    model = _build_model(args.model, env, cfg, device, agent_id=args.agent_id)
    logger = get_logger(f"{args.model}_{run.timestamp}", log_file=run.log_path)

    history = train(
        env=env, model=model,
        num_iterations=cfg["num_iterations"],
        num_episodes=cfg["num_episodes"],
        episode_length=cfg["episode_length"],
        batch_size=cfg["batch_size"],
        train_interval=cfg["train_interval"],
        logger=logger,
    )

    run.write_metrics_csv(history)

    out = {"run": run, "history": history}

    if not args.no_save_model:
        model.save(run.model_path)
        out["model_path"] = run.model_path
        print(f"saved model: {run.model_path}")

    if not args.no_plot:
        paths = plot_training(history, fig_dir=run.figs_dir, prefix="", algo_label=args.model.upper())
        out["fig_paths"] = paths
        for p in paths:
            print(f"saved figure: {p}")

    if not args.no_test_after_train and not args.no_save_model:
        # 训练完自动跑一个测试 episode，产物放在 run/test/
        model.epsilon = model.epsilon_min
        max_steps = args.max_steps if args.max_steps is not None else cfg["test_max_steps"]
        test_stats = test(env, model, max_steps=max_steps, out_dir=run.test_dir)
        out["test_stats"] = test_stats

    # 生成 summary.md 骨架
    final_stats = _collect_final_stats(history, out.get("test_stats"))
    run.write_summary_stub(final_stats=final_stats)
    print(f"summary stub: {run.summary_path}")

    return out


def _collect_final_stats(history: dict, test_stats: dict = None) -> dict:
    """把关键数字压成一维 dict 供 summary.md 填充。"""
    stats = {}
    ret = history.get("return_list") or []
    if ret:
        last10 = ret[-10:]
        stats["last10_avg_return"] = sum(last10) / len(last10)
        stats["max_return"] = max(ret)
    ber = history.get("ber_list") or []
    if ber:
        stats["last10_avg_neg_log_ber"] = sum(ber[-10:]) / min(10, len(ber))
    if test_stats:
        stats["test_success"] = test_stats.get("success")
        stats["test_steps"] = test_stats.get("steps")
        stats["test_total_reward"] = test_stats.get("total_reward")
        stats["test_mean_neg_log_ber"] = test_stats.get("mean_neg_log_ber")
    return stats


def _run_test(args, cfg: dict, env, device: torch.device) -> dict:
    run = _make_run(args, cfg, env, mode="test")
    print(f"[run] {run.name}")
    print(f"[run] dir = {run.dir}")

    model = _build_model(args.model, env, cfg, device, agent_id=args.agent_id)

    # 解析 model_path：命令行优先；否则回落到 models/<algo>_<K>agents_model.pth
    model_path = args.model_path
    if model_path is None:
        fallback = os.path.join(_ROOT, "models", f"{args.model}_{env.num_agents}agents_model.pth")
        if os.path.isfile(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"--model_path not given and fallback missing: {fallback}\n"
                "train first, or point --model_path to a run's model.pth."
            )
    model.load(model_path)
    model.epsilon = model.epsilon_min
    print(f"loaded model: {model_path}")

    max_steps = args.max_steps if args.max_steps is not None else cfg["test_max_steps"]
    test_stats = test(env, model, max_steps=max_steps, out_dir=run.test_dir)

    # test-only run 也写一个 summary 骨架
    run.write_summary_stub(
        final_stats={
            "test_success": test_stats.get("success"),
            "test_steps": test_stats.get("steps"),
            "test_total_reward": test_stats.get("total_reward"),
            "test_mean_neg_log_ber": test_stats.get("mean_neg_log_ber"),
        },
        notes=f"test-only run; loaded model: {model_path}",
    )
    return {"run": run, "test_stats": test_stats}


def main():
    # 将 Agg-setting 模块的延迟导入声明为 module-level globals，使各 helper 函数可见
    global MultiRobotEnv, DQN, MADQN, SharedMADQN, QMIX
    global train, test, plot_training, get_logger, RunContext

    args = _parse_args()

    # --- Step 1: 若 --placement custom，在任何 Agg-setting 导入之前完成交互选点 ---
    _custom_starts = _custom_targets = None
    if args.placement == "custom":
        from config.customer_choice import choose_placements
        _early_cfg = get_env_config()
        _, _custom_starts, _custom_targets = choose_placements(_early_cfg)

    # --- Step 2: 延迟导入（env.py / plot.py 此时才锁定 Agg 后端）---
    from env.env import MultiRobotEnv
    from rl_algorithms import DQN, MADQN, SharedMADQN, QMIX, train, test, plot_training
    from utils.logger_handler import get_logger
    from utils.run_manager import RunContext

    # --- Step 3: 正常构建环境 ---
    cfg = _resolve_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = get_env_config()

    # 应用自定义起终点（覆盖 scenario.npz 中的默认值）
    if _custom_starts is not None:
        env_cfg["start_states"] = _custom_starts
        env_cfg["target_states"] = _custom_targets

    # reset 起终点是否随机化：
    #   - auto（默认）：仅 shared_madqn 开启，其它算法保持 scenario.npz 固定起终点
    #   - true / false：用户手动覆盖（调试用）
    if args.randomize_reset == "auto":
        randomize = (args.model == "shared_madqn")
    else:
        randomize = (args.randomize_reset == "true")
    print(f"[env] randomize_on_reset = {randomize} ({'auto' if args.randomize_reset == 'auto' else 'override'})")
    if args.placement == "custom" and randomize:
        print("[placement] WARNING: randomize_on_reset=True — custom placement applies to first episode only.")

    # CLI 覆盖 comm_model（不传则保持 channel.yml 默认值）
    if args.comm_model is not None:
        env_cfg["comm_model"] = args.comm_model

    env = MultiRobotEnv(env_cfg, randomize_on_reset=randomize)

    if args.mode == "train":
        _run_train(args, cfg, env, device, env_cfg)
    else:
        _run_test(args, cfg, env, device)


if __name__ == "__main__":
    main()
