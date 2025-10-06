import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

crop_dir = "your path"
ave_embedding_dir = "your path"
output_dir = "your path"

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {output_dir}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    exit(1)

embedding_files = []
for root, _, files in os.walk(crop_dir):
    for file in files:
        if file.startswith("final_embedding_") and file.endswith(".txt"):
            embedding_files.append(os.path.join(root, file))
print(f"Found {len(embedding_files)} embedding files: {embedding_files}")

ave_embedding_files = [
    os.path.join(ave_embedding_dir, f) for f in os.listdir(ave_embedding_dir)
    if f.endswith(".txt")
]
print(f"Found {len(ave_embedding_files)} average embedding files: {ave_embedding_files}")

if not embedding_files or not ave_embedding_files:
    print("Error: No embedding files or average embedding files found.")
    exit(1)

results = []

for emb_path in embedding_files:
    try:
        embedding = np.loadtxt(emb_path, delimiter=',')
        print(f"Loaded embedding from {emb_path}, shape: {embedding.shape}")

        max_similarity = -1
        closest_ave_emb = None

        for ave_emb_path in ave_embedding_files:
            try:
                ave_embedding = np.loadtxt(ave_emb_path, delimiter=',')
                print(f"Loaded average embedding from {ave_emb_path}, shape: {ave_embedding.shape}")

                # 确保维度一致
                if embedding.shape != ave_embedding.shape:
                    print(f"Shape mismatch: {emb_path} ({embedding.shape}) vs {ave_emb_path} ({ave_embedding.shape})")
                    continue

                # 确保 embedding 是二维数组
                embedding_2d = embedding.reshape(1, -1)
                ave_embedding_2d = ave_embedding.reshape(1, -1)
                similarity = cosine_similarity(embedding_2d, ave_embedding_2d)[0][0]
                print(
                    f"Similarity between {os.path.basename(emb_path)} and {os.path.basename(ave_emb_path)}: {similarity}")

                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_ave_emb = os.path.basename(ave_emb_path)

            except Exception as e:
                print(f"Error processing {ave_emb_path}: {e}")
                continue

        if closest_ave_emb:
            emb_name = os.path.splitext(os.path.basename(emb_path))[0].replace("final_embedding_", "")
            ave_emb_name = os.path.splitext(closest_ave_emb)[0]
            category = "scratches" if "scratches" in ave_emb_name.lower() else "stain"
            results.append({
                "name": emb_name,
                "category": category,
                "similarity": max_similarity
            })
        else:
            print(f"No valid similarity found for {emb_path}")

    except Exception as e:
        print(f"Error processing {emb_path}: {e}")
        continue

# 保存结果到 CSV
if results:
    output_path = os.path.join(output_dir, "classification_results.csv")
    df = pd.DataFrame(results)
    try:
        df[['name', 'category', 'similarity']].to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print("Results preview:")
        print(df[['name', 'category', 'similarity']])
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
else:
    print("No results to save. Check input files and paths.")