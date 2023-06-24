# Conditionally Adaptive Multitask Learning

This repo is the implementation of the semester project "Conditionally Adaptive Multitask Learning" at [IVRL](https://www.epfl.ch/labs/ivrl/) EPFL.

## Introduction

Multitask Learning (MTL) has emerged as a framework to deal with the problems single networks per task approach presents. These problems include ignoring useful information from training signals of different tasks and memory inefficiency. MTL helps these problems by optimizing for more than one objective and brings advantages such as better generalization and reduced memory footprint. In this work, we extend the traditional multitask mechanism with conditionally adaptive modules, which produce different sets of parameters for different tasks in a multitasking network to achieve more efficient parameter sharing. Thesemodules include conditional attention, conditional adapter and conditional layer normalization.



## Main Results 

For all the models, [Swin Transformer](https://github.com/microsoft/Swin-Transformer) pre-trained with ImageNet-22K dataset is used as the base model. Our model is finetuned on top of this model using the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset. This table shows the various settings and the metrics for semantic segmentation (Mean IOU) and depth estimation (d1). More information on these metrics can be found on the [project report]().

| Setting | Multitask | ConditionedModules |Layer Normalization| Frozen Encoder | Adapter | Task Weighting |d1     | Mean <br>IOU | Model Path|
|------|--------|-----|----|----|--------|-----|---|---------|-----|
| Only Depth |False | None | Regular | False | False | -                     | 0.7588  | -       | [drive link](https://drive.google.com/file/d/1UuBHPH2v2IyGDhUtj-IK1yxcYkcTuc2H/view?usp=sharing)
| Only Segmentation |False | None | Regular | False | False | -                     | - | 0.4907    | [drive link](https://drive.google.com/file/d/1GDCWNq_V3TkVwfk0zkzDcNRskkrz99fX/view?usp=sharing)
| Baseline Multitask |True | None | Regular | False | False | 5-1                    | 0.7793 | 0.4977        |
| Frozen Encoder |True | 0-0-6-2 | Conditioned | True | True | 5-1                     | 0.5524  | 0.2666        |
| Conditioned Multitask |True | All | Conditioned | False | False | -                     | 0.7826  | 0.4978       | 
| Conditioned Multitask2 |True | All | Regular | False | False | -                     | 0.787  | 0.500       | [drive link](https://drive.google.com/file/d/1iPU1pGttKI6djgKMkGSmUxP5u-UOv6uS/view?usp=sharing)
| Taskonomy |True | All | Regular | False | False | -                     | 0.7971  | 0.6749       | [drive link](https://drive.google.com/file/d/1YqYgxqQnf9jYGB3S8r2ARSUedwOLvQYI/view?usp=sharing)

Note: Taskonomy dataset randomly selects subset of validation set to validate, therefore may get different results. Also, as they are not evaluated on the same dataset, results of Taskonomy are not directly comparable to others.

Settings represent:
- Multitask: Whether the model is a multitask model trained jointly for multiple tasks or a separate network handling only one task. Possible choices: True, False
- ConditionedModules: Swin Transformer consists of multiple transformer blocks divided into stages. In the small configuration which we use, there are 24 of these blocks, 2 in the first, 2 in the second, 18 in the third, and again 2 in the fourth stage. We experiment with various configurations of conditioning these modules. This configuration parameter represents these choices. All,None, or some of these blocks can be conditioned where the last one is shown with number of conditioned transformer blocks for each stage, for example: "0-0-18-2" conditions only the last two stages. Possible choices: All, None, 0-0-12-0
- Layer Normalization: Whether a regular or a conditioned layer normalization is used. Possible choices: Regular, Conditioned
- Frozen Encoder: Whether the encoder is open for training or not. Possible choices: True, False
-  Adapter: Whether there is adapters added to the model. Possible choices: True, False
 - Task Weighting: The hyperparameter used to balance the task weights. It has the form semantic segmentation weight - depth estimation weight. Possible choices: 5-1

More information can be found on the [project report]().

## Experiment

### Setup

The Docker file used to run the code can be found in the [https://ic-registry.epfl.ch](https://ic-registry.epfl.ch) with the name `ic-registry.epfl.ch/sinergia/sinergia_torch`. 

For all the training Nvidia A100 with 40GB memory is used.
 
-  Install the dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Training NYUv2

-  Download the data from this [Drive link](https://drive.google.com/file/d/12hWuqqcgw9BNzIhU7AIsVVv1Gj3IV5KI/view?usp=sharing) (only accesible to EPFL accounts) and put it inside the folder `data` inside a folder named `nyuv2`, with the following setup. The images are converted to .pt files for faster read.

    ```bash
    data
    └── nyuv2
        ├── combined
        ├── train_rgb_pt
        └── test_rgb_pt
    ```

- Download the pretrained Swin Transformer from the original repository. It has the Swin-S ImageNet-22K config and can be found [here](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models). Put this file under the `pretrained` folder.

    ```bash
    pretrained
    └── nyuv2
        └── swin_small_patch4_window7_224_22k.pth
    ```

#### Multitask

- `configs/conditionally_adaptive.json` file contains the configuration for the above *Conditioned Multitask2* model. It can be changed for different configuration settings.

- To train a new multitask network using the Nyuv2 Dataset, run the following code.

  ```bash
    python train_nyu.py --config ./configs/conditionally_adaptive.json
  ```

#### Single Task

- `configs/single_task.json` file contains the configuration for the single task models. `task` parameter can get *depth* or *segmentation* for different tasks. It can be changed for different configuration settings.

- To train a new multitask network using the Nyuv2 Dataset, run the following code.

  ```bash
    python train_nyu_single_task.py --config ./configs/single_task.json
  ```


- Trained model is saved to the root folder. Tensorboard outputs can be found in the `runs` folder. To open the tensorboard server, run the following command and navigate to `localhost:6007/`:
    ```bash
    tensorboard --logdir=./runs --port=6007
    ```

### Training Taskonomy

To train the model with the taskonomy dataset, get access to data. 

- `configs/taskonomy.json` file contains the configuration for the above *Conditioned Multitask2* model. It can be changed for different configuration settings. The batch size is selected for 4 Nvidia A100 GPUs. From experience, we observed 16*number of GPUs works in this setting.

- To train a new multitask network using the Nyuv2 Dataset, run the following code.

  ```bash
    python train_taskonomy.py --config ./configs/taskonomy.json
  ```

- Trained model is saved to the root folder. Tensorboard outputs can be found in the `runs` folder. To open the tensorboard server, run the following command and navigate to `localhost:6007/`:
    ```bash
    tensorboard --logdir=./runs --port=6007
    ```

### Evaluation

The trained models can be evaluated with the `evaluate.py` script.

## References

Most of the code in this repository is adapted from the following repositories. If any other code is used, a reference is given in the code.

- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [Conditional Adaptive Multi-Task Learning: a Hypernetwork for NLU](https://github.com/CAMTL/CA-MTL)