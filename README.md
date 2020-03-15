# Lane_Segmentation
FCN, DeepLab V3+, U-Net for lane segmentation in PyTorch.

## News
<li>[3/15/2020] ***Implemented Deformable Convolution***

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
python test.py
```

## Show
![Test image](https://github.com/Joyako/Lane_Segmentation/blob/master/data/test2.jpg)

