Label Enhancement for Better Model Generalization and Calibration(Under Review at CVPR2022)
================================

Concept
-----------
![concept](./img/concept.png)
* The difference between the supervision using the hard label (blue box) and the supervision using the soft label (red box)
for image classification. An example of a label enhancement for the cat class at the left is shown, and the graph at the right shows a
comparison of the traditional hard label-based and label enhancement-based classification accuracy and ECE.

Dataset
-------
* Image dataset
  * CIFAR10 (torchvision)
  * CIFAR100 (torchvision)
  * STL10 (torchvision)
  * ImageNet (https://image-net.org/)


Algorithm 
---------
* Default-Method
    * Vanilla (1)
    * Label smoothing (2)
    * Knowledge distillation[1] (3)
    
* Deterministic Label Enhancement (4, 5, 6)

 We simplified the problem by using the KD configuration by the teacher model as the enhanced label to extract feasible solutions for each sample pair (x<sub>i</sub>, z<sub>i</sub>
; θ).   
The cross entropy loss for training with the enhanced label generated bythe teacher θ<sub>t</sub> and the student model to be trained is θ<sub>s</sub> soft label (SL, #4), soft label with teacher ensemble (EN-SL, #5), KD with soft label from multiple teachers (KD-SL, #6)

* Stochastic Label Enhancement (7, 8, 9)

We approached the label enhancement problem by joining each
sample and label pair. We applied recent data augmentation[2]
techniques that can satisfy stochastic label enhancement.
Data augmentation applies equally to teacher models for label enhancement, but the mixed label is not used for training   
Similar to the deterministic approaches corresponding to #5 and #6 in Figure , label enhancement based on data augmentation can be easily extended.
![concept](./img/approach.png)


Option 
--------
* method
  *  Vanllia, Knowledge distillation Label_smoothing, SL, Aug-SL, etc. options to choose how to run the model.
* aug_method 
  * Choose which data augmentation(mixup, cutmix, ricap) to run model
* smart_model, student_model 
  * Select the model of student and teacher model (ex. ResNet20, ResNet110)

Scheduler 
---------
default schedulering (CIFAR10, 100, STL10)
* Learning rate : 0.05
* Weight decay : 5e-4
* Batch size : 64
* Epoch : 240
* learning rate scaling : 150, 180, and 210 epochs

Long training (CIFAR10, 100, STL10)
* Epoch : 350
* learning rate scaling : 150, 200, 250 and 300 epochs

default schedulering (ImageNet)
* Learning rate : 0.1
* Weight decay : 1e-4
* Batch size : 256
* Epoch : 100
* learning rate scaling : 30, 60, and 90 epochs

Training
--------
* SL
```
python main.py --smart_model_path /pretrained/models/resnet110 --is_smart_model --smart_model resnet110 --student_model resnet20 --result ./results --dataset cifar100 --n_class 100 --epochs 240 --lr_decay_epochs=150,180,210 --method sl
```
* Aug-SL+
```
python main.py --smart_model_path /pretrained/models/resnet110 --is_smart_model --smart_model resnet110 --student_model resnet32 --result ./results --dataset cifar100 --n_class 100 --epochs 350 --lr_decay_epochs=150,200,250,300 --method sl --aug_method ricap
```
Performance Matric
--------
* Expected calibration error

The ECE estimate the confidence of the neural networks.
This metric divides the confidence space into several bins and measures the observed accuracy in each bin. The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin

Experiment
--------
### Comparison results of a label enhancement experiment on the CIFAR100 dataset

| Teacher 	|      Method     	| Student(# parm) 	|                 	|                 	|                 	|                 	|
|:-------:	|:---------------:	|:---------------:	|:---------------:	|:---------------:	|:---------------:	|:---------------:	|
|         	|                 	| ResNet20(0.27M) 	| Resnet32(0.36M) 	| ResNet44(0.66M) 	| ResNet56(0.85M) 	| ResNet110(1.7M) 	|
|         	|     Vanilla     	|  69.32 / 0.070 	|  71.17 / 0.094  	|  72.11 / 0.109 	|  72.28 / 0.123  	|   73.88 / 0.131  	|
|         	| Label smoothing 	|  69.43 / 0.053 	|  71.49 / 0.032  	|  72.32 / ***0.018***  	|  72.87 / ***0.020*** 	|   73.90 / 0.051   	|
|         	|  KD (α=0.1,T=3) 	|  69.07 / 0.071  	|  71.23 / 0.092 	|  72.19 / 0.108  	|  72.76 / 0.118  	|   73.60 / 0.135  	|
|         	|        SL       	|  69.09 / 0.051  	|  71.96 / 0.071 	|  72.79 / 0.084 	|  73.59 / 0.085 	|  75.47 / 0.089 	|
|         	|      Aug-SL     	|  68.9 / 0.063  	|  72.17 / 0.060 	|  73.79 / 0.037 	|  74.87 / 0.025 	|  76.56 / 0.039 	|
|         	|     Aug-SL+     	|  69.41 / 0.073 	|  72.95 / 0.054 	|   74.30 / 0.048  	|  ***75.54*** / 0.028 	|  ***77.28*** / 0.025 	|
|         	|      En-SL      	|  70.57 / 0.050 	|  73.00 / 0.065  	|  73.84 / 0.072 	|  74.66 / 0.072 	|  76.39 / 0.071  	|
|         	|      KD-SL      	|  ***70.75*** / ***0.019*** 	|  ***73.24*** / ***0.020*** 	|  74.39 / 0.023  	|  74.84 / 0.023 	|  76.35 / ***0.016***  	|
|         	|    Aug-En-SL    	|  69.57 / 0.069 	|  72.64 / 0.059 	|  73.67 / 0.049 	|  74.66 / 0.055 	|  76.08 / 0.061 	|
|         	|    Aug-KD-SL    	|  69.13 / 0.082 	|  72.23 / 0.085 	|   73.51 / 0.080  	|  74.29 / 0.069  	|  76.04 / 0.088 	|
|         	|    Aug-En-SL+   	|  70.06 / 0.071  	|  72.94 / 0.058  	|  ***74.47*** / 0.059 	|  75.14 / 0.069 	|  76.86 / 0.074 	|
|         	|    Aug-KD-SL+   	|  70.02 / 0.084 	|  72.58 / 0.102  	|   74.20 / 0.086  	|  75.28 / 0.091  	|  76.48 / 0.089 	|

* Classification accuracy (%) and calibration scores (ECE) for ResNet of network architecture for CIFAR100. We omit variation in
ECE, which are not significant differences. All training were performed on 3 different random seeds.

### Comparison with other knowledge transfer methods.
|     Teacher    	|      ResNet110      	|      ResNet110     	|
|:--------------:	|:-------------------:	|:------------------:	|
|     Student    	|       ResNet56      	|      ResNet110     	|
| Vanilla        	|  72.28±0.09 / 0.123 	| 73.88±0.15 / 0.131 	|
| LS             	|  72.87±0.17 / ***0.020*** 	| 73.90±0.16 / 0.051 	|
| KD(α=0.1,T=3)  	| 72.76±0.11 / 0.118  	| 73.60±0.16 / 0.135 	|
| CRD [5]           	| 74.82±0.11 / 0.107  	| 76.04±0.04 / 0.121 	|
| CRD+KD [5]        	|  75.38±0.27 / 0.127 	| 76.67±0.46 / 0.125 	|
| SSKD [4]           	|  74.88±0.14 / 0.119 	| 75.73±0.17 / 0.118 	|
| WSL [6]            	|  75.08±0.19 / 0.133 	| 76.00±0.52 / 0.130 	|
| Our best (acc) 	|  ***75.54***±0.24 / 0.029 	| ***77.28***±0.20 / 0.025 	|
| Our best (ECE) 	|  74.84±0.04 / 0.023 	| 76.35±0.18 / ***0.016*** 	|
* Classification accuracy and ECE of the large student network and the performance of state-of-the-art KDs for CIFAR 100
dataset

 
References
 ----------------
* [1] Hinton et al. - ["Distilling the knowledge in a neural network"](https://arxiv.org/abs/1503.02531) (NIPSW, 2014)
* [2] Ryo Takahashi et al. - ["Data augmentation using random image cropping and patching for deep cnns"](https://arxiv.org/abs/1811.09030) (IEEE Transactions on Circuits and Systems for Video Technology, 2019)
* [3] Chuan Guo et al. - ["On calibration of modern neural networks"](https://arxiv.org/abs/1706.04599) (ICML, 2017)
* [4] Guodong Xu et al. - ["Knowledge distillation meets self-supervision"](https://arxiv.org/abs/2006.07114) (ECCV, 2020)
* [5] Yonglong Tian et al. - ["Contrastive representation distillation"](https://arxiv.org/abs/1910.10699) (ICLR 2020).
* [6] Helong Zhou et al. - ["Rethinking soft labels for knowledge distillation: A bias-variance tradeoff perspective"](https://arxiv.org/abs/2102.00650) (ICLR 2021)