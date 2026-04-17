"""
主入口：选择模型（dqn / madqn）进行训练或测试。
用法: python main.py --model madqn --mode train
      python main.py --model madqn --mode test
      python main.py --model dqn --mode test --max_steps 300
"""
import argparse
from rl_algorithms import train, test


def main():
    parser = argparse.ArgumentParser(description="DQN/MADQN 训练与测试")
    parser.add_argument("--model", choices=["dqn", "madqn"], default="madqn", help="模型: dqn 或 madqn")
    parser.add_argument("--mode", choices=["train", "test"], default="test", help="模式: train 或 test")
    parser.add_argument("--model_path", type=str, default=None, help="测试时模型路径，默认 models/<model>_model.pth")
    parser.add_argument("--max_steps", type=int, default=None, help="测试时最大步数（默认从 rl.yml 读取）")
    args = parser.parse_args()

    if args.mode == "train":
        result = train(algo=args.model)
        print(f"训练完成。模型保存至: {result.get('model_path', '未保存')}")
    else:
        result = test(algo=args.model, model_path=args.model_path,
                      max_steps=args.max_steps, save_results=True)
        if result.get("gif_path"):
            print(f"GIF: {result['gif_path']}")
        if result.get("png_path"):
            print(f"PNG: {result['png_path']}")


if __name__ == "__main__":
    main()
