import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def load_embedding(file_path):
    try:
        embedding = np.loadtxt(file_path)
        if embedding.shape != (2048,):
            print(f"警告：{file_path} 的维度为 {embedding.shape}，预期为 (2048,)，跳过")
            return None
        print(f"已加载：{file_path}")
        return embedding
    except Exception as e:
        print(f"错误：加载{file_path}时发生错误：{str(e)}")
        return None


def load_embeddings(directory):
    embeddings = []
    file_names = []

    valid_extension = '.txt'
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录 {directory} 不存在")

    for file_name in os.listdir(directory):
        if file_name.lower().endswith(valid_extension):
            file_path = os.path.join(directory, file_name)
            embedding = load_embedding(file_path)
            if embedding is not None:
                embeddings.append(embedding)
                file_names.append(Path(file_name).stem)

    return np.array(embeddings), file_names


def load_label_class(label_path):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            print(f"警告：{label_path} 为空")
            return None
        parts = lines[0].strip().split()
        if len(parts) < 1:
            print(f"警告：{label_path} 格式错误")
            return None
        return int(parts[0])
    except Exception as e:
        print(f"错误：读取{label_path}时发生错误：{str(e)}")
        return None


def get_true_category(file_name, label_dir, class_map):
    img_name = '_'.join(file_name.split('_')[:-2])
    label_path = os.path.join(label_dir, f"{img_name}.txt")

    if not os.path.exists(label_path):
        print(f"错误：未找到{img_name}对应的标注文件：{label_path}")
        return None

    class_id = load_label_class(label_path)
    if class_id is None:
        return None

    return class_map.get(class_id, None)


def save_average_embedding(average_embedding, output_dir, output_filename):
    if average_embedding is None:
        print(f"无法保存平均embedding到{output_dir}：无有效embedding")
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    try:
        # 获取保存前的文件修改时间（如果文件存在）
        original_mtime = os.path.getmtime(output_path) if os.path.exists(output_path) else 0

        # 保存embedding
        np.savetxt(output_path, average_embedding, fmt='%.6f')

        # 验证文件是否更新
        if os.path.exists(output_path):
            new_mtime = os.path.getmtime(output_path)
            if new_mtime > original_mtime:
                print(f"已成功覆盖保存平均embedding到：{output_path}")
            else:
                print(f"警告：{output_path} 未更新，可能写入失败")
        else:
            print(f"错误：{output_path} 未生成")
    except PermissionError:
        print(f"错误：无权限写入{output_path}")
    except Exception as e:
        print(f"错误：保存{output_path}时发生错误：{str(e)}")


def main():
    # 定义目录
    embeddings_dir = "your path"
    avg_embeddings_dir = "your path"
    label_dir = "your path"

    # 定义类别映射
    class_map = {0: 'scratches', 1: 'stain'}
    class_to_idx = {'scratches': 0, 'stain': 1}

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 加载平均embedding作为初始W矩阵
        avg_embeddings = {
            'scratches': load_embedding(os.path.join(avg_embeddings_dir, "scratches_average_embedding.txt")),
            'stain': load_embedding(os.path.join(avg_embeddings_dir, "stain_average_embedding.txt"))
        }
        if avg_embeddings['scratches'] is None or avg_embeddings['stain'] is None:
            print("错误：无法加载平均embedding文件")
            return

        # 构造初始W矩阵 (2 x 2048)
        W = np.vstack([avg_embeddings['scratches'], avg_embeddings['stain']])
        W = torch.tensor(W, dtype=torch.float32, requires_grad=True, device=device)

        # 初始化b向量 (2,)
        b = torch.zeros(2, dtype=torch.float32, requires_grad=True, device=device)

        # 加载待分类的embedding
        embeddings, file_names = load_embeddings(embeddings_dir)
        if len(embeddings) == 0:
            print("错误：S_Embeddings 目录中未找到有效embedding文件")
            return

        # 构造X矩阵 (2048 x N)
        X = torch.tensor(embeddings.T, dtype=torch.float32, device=device)

        # 获取真实标签
        labels = []
        valid_indices = []
        for i, file_name in enumerate(file_names):
            true_category = get_true_category(file_name, label_dir, class_map)
            if true_category is None:
                print(f"跳过 {file_name}：无法获取真实类别")
                continue
            labels.append(class_to_idx[true_category])
            valid_indices.append(i)

        if not valid_indices:
            print("错误：未找到任何有效标签")
            return

        # 过滤有效的数据
        X = X[:, valid_indices]
        file_names = [file_names[i] for i in valid_indices]
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([W, b], lr=0.01)

        # 优化循环
        max_epochs = 1000
        target_accuracy = 1.0
        for epoch in range(max_epochs):
            optimizer.zero_grad()

            # 前向传播：P = softmax(WX + b)
            logits = W @ X + b.unsqueeze(1)  # (2 x N)
            logits = logits.T  # 转置为 (N x 2)
            P = torch.softmax(logits, dim=1)

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 评估分类准确率
            with torch.no_grad():
                predictions = torch.argmax(P, dim=1)
                correct = (predictions == labels).sum().item()
                accuracy = correct / len(labels)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.4f}")

                if accuracy >= target_accuracy:
                    print(f"达到目标准确率 {accuracy * 100:.2f}% 在 epoch {epoch}")
                    break

        # 使用优化后的W进行最终分类
        with torch.no_grad():
            logits = W @ X + b.unsqueeze(1)
            logits = logits.T  # 转置为 (N x 2)
            P = torch.softmax(logits, dim=1)
            predictions = torch.argmax(P, dim=1)

        # 打印分类结果
        print("\n分类结果：")
        print("Embedding Name".ljust(30) + " | Classified Category | Correctness")
        print("-" * 80)
        correct_count = 0
        for i, file_name in enumerate(file_names):
            predicted_category = 'scratches' if predictions[i].item() == 0 else 'stain'
            true_category = class_map[labels[i].item()]
            correctness = 'Correct' if predicted_category == true_category else 'Incorrect'
            if correctness == 'Correct':
                correct_count += 1
            print(f"{file_name:<30} | {predicted_category:<18} | {correctness}")

        accuracy = correct_count / len(file_names) * 100
        print(f"\n已处理 {len(file_names)} 个embedding")
        print(f"最终分类准确率：{accuracy:.2f}% ({correct_count}/{len(file_names)} 正确)")

        # 保存优化后的W矩阵
        W_np = W.cpu().numpy()
        save_average_embedding(W_np[0], avg_embeddings_dir, "scratches_average_embedding.txt")
        save_average_embedding(W_np[1], avg_embeddings_dir, "stain_average_embedding.txt")

    except FileNotFoundError as e:
        print(f"错误：{str(e)}")
    except Exception as e:
        print(f"错误：发生意外错误：{str(e)}")


if __name__ == "__main__":
    main()