# Few-shot image classification for defect detection in aviation materials

by Zewei Wu and Chengbin Peng, details are in paper.

This study proposes a learning approach that can be trained using standardized industrial defect datasets to detect defects in few-shot non-standardized industrial defect production. Specifically, this approach initially utilizes the knowledge acquired from a standardized pretraining dataset to crop the data within the query set. Subsequently, an embedding extraction is employed to obtain feature embedding and classification results. A Faster R-CNN detector and a DINOv2 embedding extractor are used. A single-file model package and a Flask web interface are provided for inference.

### Dataset:

This dataset of surface defects in civil aviation steel components is split into a support set and a query set and includes two defect categories: scratches and stains.

### Requirements:

- environment:
  
  ```
  Windows 10/11
  NVIDIA GeForce 3060 GPU (CPU fallback if unavailable)
  Python 3.9
  MySQL 8 (localhost:33061, database AFDN)
  ```

- python packages:
  
  ```
  torch 2.0.1
  torchvision 0.15.2
  timm 0.9.16
  numpy 1.23.2
  opencv-python 4.6.0.66
  Pillow 9.2.0
  pandas 1.5.0
  scikit-learn 1.1.2
  Flask 3.1.3
  PyMySQL 1.2.0
  openpyxl 3.1.5
  ```

### Train

```
train_Neu_faster_rcnn.py
train_Boeing_faster_rcnn.py
vgg16_train_predict.py
protonet_train_predict.py
```

### Embedding

```
GetEmbedding.py
Ave_Embedding.py
CAE.py
Recommend.py
AFDNClassify.py
```

Run the offline pipeline step by step:

```
crops.py
GetEmbedding.py
Ave_Embedding.py
CAE.py
Recommend.py
AFDNClassify.py
```

or run all steps at once:

```
run_pipeline.py
```

### Web

Build the single-file model package and start the Flask application:

```
build_model_package.py
py -3 web/app.py
```

Then open http://127.0.0.1:5000 to upload one or more images for defect classification. Results include detection boxes and can be exported to Excel.

### Citation:

```
 @article{117749,
  title={Few-shot image classification for defect detection in aviation materials},
  author={Wu, Zewei and Peng, Chengbin},
  journal={Measurement},
  year={2025},
  keywords={Defect classification, Image Embedding, Domain adaptation, Few-shot learning}
}
```
