"""使用 Agent 策略的随机测试与渲染。"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.env import Env
from rl_algorithms.utils.agent import Agent


def run_with_agent(env, agents, max_steps=200, debug=False):
    if not isinstance(agents, (list, tuple)):
        agents = [agents for _ in range(env.num_agents)]
    elif len(agents) != env.num_agents:
        raise ValueError(f"传入的Agent数量({len(agents)})与环境配置的num_agents({env.num_agents})不一致")

    states, _ = env.reset()
    print(f"Agent数量: {env.num_agents}")
    print(f"初始状态: {states}")
    print(f"目标状态: {env.target_states}")
    print("-" * 50)
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}

    for step in range(max_steps):
        actions = [agents[i].select_action(states[i] if isinstance(states, list) else states, agent_id=i, training=False)
                   for i in range(env.num_agents)]
        next_states, rewards, dones, _ = env.step(actions)
        if not isinstance(next_states, list):
            next_states, rewards, dones = [next_states], [rewards], [dones]
        if debug:
            action_str = ", ".join([f"Agent{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
            state_str = ", ".join([f"Agent{i+1}:{s}" for i, s in enumerate(next_states)])
            print(f"步数 {step+1}: 动作=[{action_str}], 状态=[{state_str}]")
        states = next_states
        if all(dones):
            print(f"\n所有Agent成功到达目标！总步数: {step+1}")
            return True
    print(f"\n达到最大步数 {max_steps}，未全部到达目标")
    return False


def main():
    print("=" * 50)
    print("使用Agent策略测试")
    print("=" * 50)
    env = Env()
    agents = [Agent(env) for _ in range(env.num_agents)]
    start_time = time.time()
    success = run_with_agent(env, agents, max_steps=200, debug=False)
    print(f"运行时间: {time.time() - start_time}秒")
    print("\n开始渲染环境")
    env.render(mode="human")
    try:
        env.render_animation(interval=1, save_path=None)
    except Exception as e:
        print(f"动画渲染出错: {e}")
        env.render(mode="human")
    print("\n渲染完成！")


if __name__ == "__main__":
    main()
