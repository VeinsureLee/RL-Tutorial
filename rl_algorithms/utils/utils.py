import numpy as np
import torch


def state_to_idx_tensor(state, y_dim, device):
    """
    将状态转换为 state idx 张量
    :param state: 单个状态 (x, y) 或批量状态数组 (batch, 2)
    :return: state idx tensor，单个状态返回形状 (1,)，批量状态返回形状 (batch,)
    """
    if isinstance(state, (tuple, list, np.ndarray)) and len(state) == 2 and isinstance(state[0], (int, np.integer)):
        # 单个状态 (x, y)
        state_idx = int(state[0]) * y_dim + int(state[1])
        state_idx_tensor = torch.tensor(
            [state_idx], dtype=torch.long, device=device)
    else:
        # 批量状态 (batch, 2)
        state = np.array(state, dtype=np.int64)
        state_indices = state[:, 0] * y_dim + state[:, 1]
        state_idx_tensor = torch.tensor(
            state_indices, dtype=torch.long, device=device)
    return state_idx_tensor


def idx_tensor_to_state(idx_tensor, y_dim):
    """
    将 state idx 张量转换为状态
    :param idx_tensor: state idx tensor，单个状态返回形状 (1,) 或标量，批量状态返回形状 (batch,)
    :param y_dim: 环境 y 维度
    :return: 状态数组，单个状态返回形状 (2,)，批量状态返回形状 (batch, 2)
    """
    # 转换为 numpy 数组以便处理
    if isinstance(idx_tensor, torch.Tensor):
        idx_array = idx_tensor.cpu().numpy()
    elif isinstance(idx_tensor, (int, np.integer)):
        # 单个标量
        idx_array = np.array([idx_tensor])
    else:
        idx_array = np.array(idx_tensor)

    # 确保是一维数组
    if idx_array.ndim == 0:
        idx_array = idx_array.reshape(1)

    # 计算状态坐标
    state_x = idx_array // y_dim
    state_y = idx_array % y_dim

    # 判断是单个状态还是批量状态
    if len(idx_array) == 1:
        # 单个状态 (x, y)
        state = np.array([int(state_x[0]), int(state_y[0])], dtype=np.int64)
    else:
        # 批量状态 (batch, 2)
        state = np.stack([state_x, state_y], axis=1).astype(np.int64)

    return state


def calculate_distance(state_idx, targets_idx, y_dim=10, x_dim=10, device='cpu'):
    """
    计算状态到目标的距离信息
    支持两种模式：
    1. 批量模式：state_idx 和 targets_idx 形状相同 (B,)，每个样本的 state 对应一个 target
    2. 多目标模式：state_idx 是单个值，targets_idx 是多个值 (num_targets,)，计算单个 state 到多个 targets 的距离
    
    :param state_idx: 状态索引，torch.Tensor 形状 (B,) 或 (1,) 或标量
    :param targets_idx: 目标索引，torch.Tensor 形状 (B,) 或 (num_targets,)
    :param y_dim: 环境 y 维度
    :param x_dim: 环境 x 维度
    :param device: 设备
    :return: 字典，包含 state_idx, targets_idx 和所有距离特征
    """
    # 确保 state_idx 和 targets_idx 是 torch.Tensor
    if not isinstance(state_idx, torch.Tensor):
        state_idx = torch.tensor([state_idx] if isinstance(
            state_idx, (int, np.integer)) else state_idx, device=device)
    if not isinstance(targets_idx, torch.Tensor):
        targets_idx = torch.tensor(targets_idx, device=device)

    # 确保在正确的设备上
    state_idx = state_idx.to(device)
    targets_idx = targets_idx.to(device)

    # 转换为状态坐标
    state = idx_tensor_to_state(state_idx, y_dim)  # 单个状态: (2,), 批量状态: (B, 2)
    # 单个目标: (2,), 批量目标: (B, 2) 或 (num_targets, 2)
    targets = idx_tensor_to_state(targets_idx, y_dim)

    # 转换为 torch tensor
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    # 判断处理模式
    if state_tensor.ndim == 1 and state_tensor.shape[0] == 2:
        # 单个 state 模式
        if targets_tensor.ndim == 1 and targets_tensor.shape[0] == 2:
            # 单个 state 到单个 target
            state_expanded = state_tensor.unsqueeze(0)  # (1, 2)
            targets_expanded = targets_tensor.unsqueeze(0)  # (1, 2)
            dx = (targets_expanded[:, 0] - state_expanded[0, 0])  # (1,)
            dy = (targets_expanded[:, 1] - state_expanded[0, 1])  # (1,)
        else:
            # 单个 state 到多个 targets
            state_expanded = state_tensor.unsqueeze(0)  # (1, 2)
            # (num_targets,)
            dx = (targets_tensor[:, 0] - state_expanded[0, 0])
            # (num_targets,)
            dy = (targets_tensor[:, 1] - state_expanded[0, 1])
    else:
        # 批量模式：state 和 target 数量相同
        dx = (targets_tensor[:, 0] - state_tensor[:, 0])  # (B,)
        dy = (targets_tensor[:, 1] - state_tensor[:, 1])  # (B,)

    # 归一化
    dx_n = dx / x_dim
    dy_n = dy / y_dim

    # 计算距离特征
    abs_dx = torch.abs(dx_n)
    abs_dy = torch.abs(dy_n)
    l1 = abs_dx + abs_dy
    l2 = torch.sqrt(dx_n ** 2 + dy_n ** 2 + 1e-8)
    sign_dx = torch.sign(dx_n)
    sign_dy = torch.sign(dy_n)

    # 返回包含 state_idx 和所有 target 距离信息的字典
    return {
        'state_idx': state_idx,
        'targets_idx': targets_idx,
        'state_coord': state_tensor,
        'targets_coord': targets_tensor,
        'dx_n': dx_n,
        'dy_n': dy_n,
        'abs_dx': abs_dx,
        'abs_dy': abs_dy,
        'l1': l1,
        'l2': l2,
        'sign_dx': sign_dx,
        'sign_dy': sign_dy,
    }


if __name__ == "__main__":
    state_test = torch.tensor([(1, 2)])
    state_test_idx = state_to_idx_tensor(state_test, 10, 'cpu')
    print(state_test_idx)
    state_idx_test = torch.tensor([12, 13, 14])
    state_test = idx_tensor_to_state(state_idx_test, 10)
    print(state_test)
    state_idx_test = torch.tensor([12])
    targets_idx_test = torch.tensor([13, 14, 15])
    distance_info = calculate_distance(state_idx_test, targets_idx_test)
    print("State idx:", distance_info['state_idx'])
    print("Targets idx:", distance_info['targets_idx'])
    print("State coord:", distance_info['state_coord'])
    print("Targets coord:", distance_info['targets_coord'])
    print("L1 distance:", distance_info['l1'])
    print("L2 distance:", distance_info['l2'])
    print("All distance features:")
    for key, value in distance_info.items():
        if key not in ['state_idx', 'targets_idx', 'state_coord', 'targets_coord']:
            print(f"  {key}: {value}")
