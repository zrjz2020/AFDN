# Few-shot image classification for defect detection in aviation materials

by Zewei Wu and Chengbin Peng, details are in paper.

This study proposes a learning approach that can be trained using standardized industrial defect datasets to detect defects in few-shot non-standardized industrial defect production. Specifically, this approach initially utilizes the knowledge acquired from a standardized pretraining dataset to crop the data within the query set. Subsequently, an embedding extraction is employed to obtain feature embedding and classification results.

### Dataset:

This dataset of surface defects in civil aviation steel components is split into a support set and a query set and includes two defect categories: scratches and stains.

### Requirements:

- environment:
  
  ```
  Windows11 24H2
  Intel Core i5-12400F processor 
  NVIDIA GeForce 3060 GPU
  DDR4 2666 MHz 32 GB RAM
  Python 3.8.8
  ```

- python packages:
  
  ```
  torch 2.0.1
  torchvision 0.15.2
  pandas 1.2.4
  ```

### Train

```
train_Boeing_faster_rcnn.py
train_Boeing_yolo.py
train_Neu_faster_rcnn.py
train_neu_yolo.py
vgg16_train_predict.py
protonet_train_predict.py
```

### Embedding

```
GetEmbedding.py
Ave_Embedding.py
CAE.py
```

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

