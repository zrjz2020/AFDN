import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

_parser = argparse.ArgumentParser(description="AFDN 少样本分类")
_parser.add_argument("--det-prior", dest="det_prior", action="store_true", default=True,
                     help="启用检测 rank-1 先验（默认启用）")
_parser.add_argument("--no-det-prior", dest="det_prior", action="store_false",
                     help="禁用检测先验，纯 embedding 原型匹配（原版逻辑）")
_parser.add_argument("--sim-threshold", type=float, default=0.6,
                     help="相似度阈值，低于此值且与检测冲突时采信检测 rank-1（默认 0.6）")
_args = _parser.parse_args()
print(f"det_prior={_args.det_prior}  sim_threshold={_args.sim_threshold}")

crop_dir = "./datasets/BoeingFewShot/crop"
_predict_root = "./runs/predict"
_exps = sorted([d for d in os.listdir(_predict_root)
                if os.path.isdir(os.path.join(_predict_root, d)) and d.startswith("exp")],
               key=lambda x: int(x.replace("exp", ""))) if os.path.isdir(_predict_root) else []
_predict_csv = (os.path.join(_predict_root, _exps[-1], "predictions_rcnn.csv")
                if _exps else None)
print(f"Predictions CSV: {_predict_csv}")
_cae_dir = "./datasets/BoeingFewShot/embeddings/CAE_AveEmbedding"
ave_embedding_dir = (
    _cae_dir
    if (os.path.isdir(_cae_dir) and any(f.endswith(".txt") for f in os.listdir(_cae_dir)))
    else "./datasets/BoeingFewShot/embeddings/AveEmbedding"
)
print(f"Prototype dir: {ave_embedding_dir}")
output_dir = "./datasets/BoeingFewShot/R"

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
print(f"Found {len(embedding_files)} embedding files")

ave_embedding_files = [
    os.path.join(ave_embedding_dir, f) for f in os.listdir(ave_embedding_dir)
    if f.endswith(".txt")
]
print(f"Found {len(ave_embedding_files)} average embedding files")

if not embedding_files or not ave_embedding_files:
    print("Error: No embedding files or average embedding files found.")
    exit(1)

import csv as _csv

_det_records = {}
if _predict_csv and os.path.isfile(_predict_csv):
    with open(_predict_csv, "r", encoding="utf-8") as _f:
        _reader = _csv.DictReader(_f)
        for _row in _reader:
            if _row.get("Status") != "Detected":
                continue
            _img = os.path.splitext(_row["Image"])[0]
            try:
                _cls = _row["Predicted Class"].strip().lower()
                _conf = float(_row["Confidence/Error"])
            except (ValueError, KeyError):
                continue
            _det_records.setdefault(_img, []).append((_cls, _conf))


def detection_prior(img_name):
    votes = {"scratches": 0, "stain": 0}
    rank1_cls = None
    rank1_conf = -1.0
    for cls, conf in _det_records.get(img_name, []):
        votes[cls] = votes.get(cls, 0) + 1
        if conf > rank1_conf:
            rank1_conf, rank1_cls = conf, cls
    return rank1_cls, votes


results = []

for emb_path in embedding_files:
    try:
        embedding = np.loadtxt(emb_path)
        emb_name = os.path.splitext(os.path.basename(emb_path))[0].replace("final_embedding_", "")

        max_similarity = -1
        closest_ave_emb = None
        for ave_emb_path in ave_embedding_files:
            try:
                ave_embedding = np.loadtxt(ave_emb_path)
                if embedding.shape != ave_embedding.shape:
                    print(f"Shape mismatch: {emb_path} vs {ave_emb_path}")
                    continue
                similarity = cosine_similarity(embedding.reshape(1, -1),
                                               ave_embedding.reshape(1, -1))[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_ave_emb = os.path.basename(ave_emb_path)
            except Exception as e:
                print(f"Error processing {ave_emb_path}: {e}")

        if not closest_ave_emb:
            print(f"No valid similarity found for {emb_path}")
            continue

        ave_emb_name = os.path.splitext(closest_ave_emb)[0]
        emb_category = "scratches" if "scratches" in ave_emb_name.lower() else "stain"

        det_category, votes = detection_prior(emb_name)
        total_votes = votes["scratches"] + votes["stain"]
        sim = float(max_similarity)
        if _args.det_prior and det_category:
            if emb_category == det_category:
                category = emb_category
            elif sim < _args.sim_threshold:
                category = det_category
            else:
                category = emb_category
        else:
            det_category = det_category or "none"
            category = emb_category

        results.append({
            "name": emb_name,
            "category": category,
            "similarity": round(float(max_similarity), 6),
            "emb_category": emb_category,
            "det_category": det_category,
            "votes_scratches": votes["scratches"],
            "votes_stain": votes["stain"],
        })

    except Exception as e:
        print(f"Error processing {emb_path}: {e}")
        continue

if results:
    output_path = os.path.join(output_dir, "classification_results.csv")
    df = pd.DataFrame(results)
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(df)
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
else:
    print("No results to save. Check input files and paths.")
