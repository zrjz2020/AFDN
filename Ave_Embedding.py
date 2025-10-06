import os
import numpy as np


def load_embeddings(directory):
    """加载目录中的所有embedding文件"""
    embeddings = []
    file_names = []

    # 支持的扩展名
    valid_extension = '.txt'

    # 检查目录是否存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录 {directory} 不存在")

    # 遍历目录中的所有.txt文件
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(valid_extension):
            file_path = os.path.join(directory, file_name)
            try:
                embedding = np.loadtxt(file_path)
                if embedding.shape != (2048,):
                    print(f"警告：{file_name} 的维度为 {embedding.shape}，预期为 (2048,)，跳过")
                    continue
                embeddings.append(embedding)
                file_names.append(file_name)
                print(f"已加载：{file_path}")
            except Exception as e:
                print(f"错误：加载{file_name}时发生错误：{str(e)}")

    return np.array(embeddings), file_names


def calculate_average_embedding(embeddings):
    if len(embeddings) == 0:
        return None
    # 沿第0轴（样本轴）计算平均值
    return np.mean(embeddings, axis=0)


def save_average_embedding(average_embedding, output_dir, output_filename):
    if average_embedding is None:
        print(f"无法保存平均embedding到{output_dir}：无有效embedding")
        return
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    try:
        np.savetxt(output_path, average_embedding, fmt='%.6f')
        print(f"已保存平均embedding到：{output_path}")
    except Exception as e:
        print(f"错误：保存{output_path}时发生错误：{str(e)}")


def main():
    # 定义两种类型的embedding目录
    directories = {
        'scratches': "your path",
        'stain': "your path"
    }
    # 定义平均embedding的输出目录
    output_dir = "your path"

    for category, directory in directories.items():
        try:
            # 加载该类别的所有embedding
            embeddings, file_names = load_embeddings(directory)

            if len(embeddings) == 0:
                print(f"错误：{category} 目录中未找到有效embedding文件")
                continue

            # 计算平均embedding
            average_embedding = calculate_average_embedding(embeddings)

            # 保存平均embedding，文件名包含类别
            output_filename = f"{category}_average_embedding.txt"
            save_average_embedding(average_embedding, output_dir, output_filename)

            print(f"已处理类别 {category}，共 {len(embeddings)} 个embedding")

        except FileNotFoundError as e:
            print(f"错误：{str(e)}")
        except Exception as e:
            print(f"错误：处理{category}时发生意外错误：{str(e)}")


if __name__ == "__main__":
    main()