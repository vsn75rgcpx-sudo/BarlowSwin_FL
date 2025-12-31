"""
test_only.py (Fixed Version)
修正了“只推理中心裁剪区域”的 Bug，启用全脑滑动窗口推理。
"""
import os
import torch
import numpy as np
import random
# 从主脚本导入必要的类和配置
from train_fednas_full import BraTSDataset, final_test_phase, infer_input_shape, CONFIG, RESULT_DIR, CHECKPOINT_DIR


def run_test():
    # 1. 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("[Test Only] Initializing...")

    # 2. 准备数据
    data_root = "dataset"
    if not os.path.exists(data_root):
        raise ValueError("dataset/ folder not found")

    all_cases = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    random.shuffle(all_cases)

    # 按照 80/20 划分
    split_idx = int(len(all_cases) * 0.8)
    test_ids = all_cases[split_idx:]
    print(f"[Test Only] Found {len(test_ids)} test cases.")

    # =====================================================
    # [核心修正 1] 开启 test_mode=True
    # 这样 dataset 会返回 (240, 240, 155) 的全图，而不是 (96, 96, 96) 的裁剪
    # =====================================================
    test_ds = BraTSDataset(data_root, test_ids, augment=False, test_mode=True)

    # 获取通道数 (通常是 4)
    in_channels, _ = infer_input_shape(test_ds)

    # =====================================================
    # [核心修正 2] 强制指定模型分辨率为 (96, 96, 96)
    # 这一步至关重要！因为权重是在 96x96x96 的模型上训练的。
    # 这里的 resolution 指的是“滑动窗口的大小”，而不是“整张图的大小”。
    # 滑动窗口机制会负责把 240 的大图切成 96 的块喂给这个模型。
    # =====================================================
    model_resolution = (96, 96, 96)

    # 架构文件路径
    json_path = os.path.join(RESULT_DIR, "best_arch.json")

    # 权重文件路径
    avg_model = os.path.join(CHECKPOINT_DIR, "retrain_averaged_model.pth")
    best_model = os.path.join(CHECKPOINT_DIR, "retrain_best_model.pth")
    model_to_load = avg_model if os.path.exists(avg_model) else best_model

    if not os.path.exists(model_to_load):
        raise FileNotFoundError(f"No model found at {avg_model} or {best_model}")

    print(f"[Test Only] Using Architecture: {json_path}")
    print(f"[Test Only] Using Weights: {model_to_load}")
    print(f"[Test Only] Inference Mode: Full Volume Sliding Window (Window Size: {model_resolution})")
    inference_weights = [1.0, 1.2, 1.0, 1.2]

    print(f"[Test] Inference Weights: {inference_weights}")
    # 4. 运行测试
    # final_test_phase 内部会调用 metrics.sliding_window_inference
    final_test_phase(
        test_dataset=test_ds,
        arch_json=json_path,
        device=CONFIG["device"],
        in_channels=in_channels,
        resolution=model_resolution,  # 传入强制的 96x96x96 分辨率 (作为窗口大小)
        best_model_path=model_to_load
    )


if __name__ == "__main__":
    run_test()