# Configuration Settings

Each of the training and evaluation config files have different parameters.

## Train NYU2 Multitask
```json
{
    "experiment_name": "conditionally_adaptive",
    "continue_training": false,
    
    "batch_size": 16,
    "epochs": 4000,
    
    "conditioned_blocks": [[0,1],[0,1],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[0,1]],
    "adapter": false,
    "frozen_encoder": false,
    "use_conditional_layer_norm": false,
        
    "seg_weight": 5,
    "depth_weight": 1
}

```
Training settings:
- experiment_name: name of the experiment (string)
- continue_training: start over or continue training with the same experiment name (boolean)
- batch_size: batch size for the training (integer)
- epochs: number of epochs (integer)
- seg_weight: Weight of segmentation (integer)
- depth_weight: Weight of depth (integer)

Model settings:
- conditioned_blocks: blocks to condition (more information on README) (list)
- adapter: whether to use adapter (boolean)
- frozen_encoder: whether to use frozen encoder (boolean)
- use_conditional_layer_norm: whether to use conditional layer normalization (boolean)


## Train NYU2 Single Task
```json
{
    "experiment_name": "single_task_segmentation",
    "continue_training": false,
    
    "task": "segmentation",
    
    "batch_size": 16,
    "epochs": 4000,
    
    "frozen_encoder": false
        
}

```
Training settings:
- experiment_name: name of the experiment (string)
- continue_training: start over or continue training with the same experiment name (boolean)
- batch_size: batch size for the training (integer)
- epochs: number of epochs (integer)

Model settings:
- task: task to train ("segmentation" or "depth")
- frozen_encoder: whether to use frozen encoder (boolean)


## Train Taskonomy
```json
{
    "experiment_name": "taskonomy_adaptive",
    "continue_training": false,
    
    "batch_size": 64,
    "epochs": 10,
    
    "conditioned_blocks": [[0,1],[0,1],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[0,1]],
    "adapter": false,
    "frozen_encoder": false,
        
    "seg_weight": 5,
    "depth_weight": 1
}


```
Training settings:
- experiment_name: name of the experiment (string)
- continue_training: start over or continue training with the same experiment name (boolean)
- batch_size: batch size for the training (integer)
- epochs: number of epochs (integer)
- seg_weight: Weight of segmentation (integer)
- depth_weight: Weight of depth (integer)

Model settings:
- conditioned_blocks: blocks to condition (more information on README) (list)
- adapter: whether to use adapter (boolean)
- frozen_encoder: whether to use frozen encoder (boolean)


## Evaluate
```json
{
    "model_path": "conditionally_adaptive.pt",
    "setting": "nyu",
    "num_generated_images": 2,
        
    "conditioned_blocks": [[0,1],[0,1],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[0,1]],
    "adapter": false,
    "frozen_encoder": false,
    "use_conditional_layer_norm": false   
}
```

- model_path: path of the model (string)
- setting: setting of the model ("nyu", "nyu_single_task" or "taskonomy")
- num_generated_images: number of images to be generated and saved (integer)

Also, based on the setting, model settings given above need to be included.
