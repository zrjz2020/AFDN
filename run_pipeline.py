import subprocess
import sys
import os
import time

STEPS = [
    ("环境自检", None),
    ("1/6 检测裁剪 (crops.py)", "crops.py"),
    ("2/6 Support 特征 (GetEmbedding.py)", "GetEmbedding.py"),
    ("3/6 类别原型 (Ave_Embedding.py)", "Ave_Embedding.py"),
    ("4/6 CAE 原型优化 (CAE.py)", "CAE.py"),
    ("5/6 Query 推荐 (Recommend.py)", "Recommend.py"),
    ("6/6 融合分类 (AFDNClassify.py)", "AFDNClassify.py"),
]


def env_check():
    import torch
    ok = torch.cuda.is_available()
    ver = torch.__version__
    print(f"torch {ver}  cuda_available={ok}  device={torch.cuda.get_device_name(0) if ok else 'cpu'}")
    if not ok:
        print("警告：CUDA 不可用，将用 CPU 跑（速度慢但结果一致）")
    assets = [
        ("Boeing R-CNN 权重", "./runs/train/exp21/weights/best.pt"),
        ("Query 图目录", "./datasets/BoeingFewShot/Q/images"),
        ("Support 图目录", "./datasets/BoeingFewShot/S/images"),
        ("DINOv2 权重", "./weights/dinov2_vitb14_pretrain.pth"),
    ]
    missing = [(n, p) for n, p in assets if not os.path.exists(p)]
    for n, p in assets:
        print(f"  {'OK ' if os.path.exists(p) else '缺失'} {n}: {p}")
    if missing:
        print("存在缺失资产，终止。")
        return False
    return True


def main():
    t0 = time.time()
    print("=" * 60)
    for name, script in STEPS:
        print(f"\n>>> {name}")
        print("-" * 60)
        if script is None:
            if not env_check():
                sys.exit(1)
            continue
        r = subprocess.run([sys.executable, "-u", script])
        if r.returncode != 0:
            print(f"\n[FAILED] {name} exit={r.returncode}，流水线中止。")
            sys.exit(r.returncode)
    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"流水线全部完成，总耗时 {dt/60:.1f} 分钟")
    print("分类结果：./datasets/BoeingFewShot/R/classification_results.csv")
    print("检测记录：./runs/predict/exp*/predictions_rcnn.csv")


if __name__ == "__main__":
    main()
