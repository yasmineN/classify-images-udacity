# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. 



In this project, first develop code for an image classifier built with PyTorch, then convert it into a command line application.



## Usage the command line application


### Basic Usage for training 
```
python train.py data_directory
```

Options

Set directory to save checkpoints
```
python train.py data_dir --save_dir save_directory
```

Choose architecture - Neural network (options : vgg16 - alexnet - resnet )
```
python train.py data_dir --arch "vgg16"
```

Set hyperparameters
```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```

Use GPU for training
```
python train.py data_dir --gpu
```

### Basic Usage for prediticting 
```
python predict.py /path/to/image checkpoint
```

Options

Return top K most likely classes
```
python predict.py input checkpoint --top_k 

```
Use a mapping of categories to real names
```
python predict.py input checkpoint --category_names cat_to_name.json
```

Use GPU for inference
```
python predict.py input checkpoint --gpu
```