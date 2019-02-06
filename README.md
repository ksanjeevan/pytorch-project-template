

# Pytorch Project Template

### Example: Image Classification

Inspired / worked off [victoresque](https://github.com/victoresque/pytorch-template).

Example using `kaggle datasets download -d alxmamaev/flowers-recognition` dataset.

#### Project Structure
```bash
├── run.py
├── config.json
├── data
│   ├── __init__.py
│   ├── data_sets.py
│   ├── data_manager.py
│   └── transforms.py
├── eval
│   ├── __init__.py
│   ├── infer.py
│   └── evaluate.py
├── net
│   ├── __init__.py
│   ├── base_model.py
│   ├── loss.py
│   ├── metric.py
│   └── model.py
├── saved
│   ├── 0203_174536
│   │   ├── checkpoints
│   │   │  
│   │   └── logs
│   │       ├── train
│   │       └── valid
│   ...
├── train
│   ├── __init__.py
│   ├── base_trainer.py
│   └── trainer.py
└── utils
    ├── __init__.py
    ├── logger.py
    ├── util.py
    ├── Verdana.ttf
    └── visualization.py
```

#### Config file explanation
```
{

    "name"          :   "PyTorch Template",
    "data"          :   {
                            "type"      :   "FolderDataManager", # Class that handles DataLoaders, splits, etc.
                            "path"      :   "/home/kiran/Documents/DATA/flowers", # Path to data
                            "format"    :   "image", # Data format (determines read function for Dataset)
                            "loader"    :   { # DataLoader arguments
                                                "shuffle"       : true,
                                                "batch_size"    : 16,
                                                "num_workers"   : 4,
                                                "drop_last"     : true
                                            },
                            "splits"    :   { # Data split %s
                                                "train" : 0.7, 
                                                "val"   : 0.2,
                                                "test"  : 0.1        
                                            }
                        },
    "transforms"    :   { # Class in charge of data augmentation + augmentation values
                            "type"      :   "ImageTransforms",
                            "args"      :   {
                                                "size"          : 224,
                                                "scale"         : [0.08, 1.0],
                                                "ratio"         : [0.75, 1.333],
                                                "colorjitter"         : [0.2,0.2,0.2]
                                            }
                        },
    "optimizer"     :   { # Class of optimizer + args
                            "type"      :   "Adam",
                            "args"      :   {
                                                "lr"            : 0.0005,
                                                "weight_decay"  : 0,
                                                "amsgrad"       : false
                                            }
                        },
    "model"         :   { # What defined model to use
                            "type"      :   "VGG16"
                        },
    "train"         :   { # Training parameters
                            "loss"      :   "cross_entropy",
                            "epochs"    :   100,
                            "save_dir"  :   "saved/",
                            "save_p"    :   1,
                            "verbosity" :   2,
                            
                            "monitor"   :   "min val_loss",
                            "early_stop":   10,
                            "tbX"       :   true
                        },
    "metrics"       :   "classification_metrics" # defined function that returns metrics to track

}
```
#### Usage
##### Training
```
./run.py train -c config_flowers.json
```
And then monitor with **TensorBoard** (will create a training / validation split):

<p align="center">
<img src="result_plots/tbx.png" width="600px"/>
</p>

To resume training given a checkpoint:
```
./run.py train -r saved/0203_180810/checkpoints/model_best.pth
```

##### Evaluating
```
./run.py eval -r saved/0203_162748/checkpoints/model_best.pth 
```
Will run evaluation metrics on the test split of the dataset:

```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:01<00:00, 17.45it/s]
{'avg_precision': 0.77, 'avg_recall': 0.79, 'accuracy': 0.81}

```

##### Inference
Given a trained model:
```
./run.py result_plots/daisy.jpeg -r saved/0203_162748/checkpoints/model_best.pth 
```

 Using a pretrained model:
```
./run.py result_plots/bird.jpeg -r saved/0203_180810/checkpoints/model_best.pth
```
 
Respectively:

<p align="center">
<img src="result_plots/daisy_pred.png" width="250px"/>
<img src="result_plots/bird_pred.png" width="293px"/>
</p>

#### Notes
If using ***FolderDataManager***, a file called *.splits.json* will be created in the directory of the chosen dataset. This is so after randomization, the train/val/test splits on multiple runs remain the same and the performance can be properly benchmarked. Simply delete *.splits.json* in order to reshuffle the splits.





