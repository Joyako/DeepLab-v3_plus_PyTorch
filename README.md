# Lane_Segmentation
FCN, DeepLab V3+, U-Net for lane segmentation in PyTorch.

## Dataset & Training
To train the network, one can use [competition data set](https://aistudio.baidu.com/aistudio/competition/detail/5).
Firstly, Downloading it. The Dataset with 21914 images, the number of training, validation and test sets is 15339,2192,4383 respectively. 

Then,  run to train:
```base
python train.py
```

## Evaluation
Run:
```bash
python ./test/test_lane_segmentation.py
```

