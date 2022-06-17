# Confusion Matrix


### Installation

`conda env create -f confusion_matrix.yml` \
`conda activate confusion-matrix`

### Run

`python3 main.py --input 'data/input' --target 'data/target' --predict 'data/predict' --write_csv --figure --norm_conf`

### Help

```
usage: main.py [-h] [--input INPUT] [--target TARGET] [--predict PREDICT]
               [--num_classes NUM_CLASSES] [--results RESULTS] [--norm_conf]
               [--figure] [--write_csv]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to input images
  --target TARGET       path to target images
  --predict PREDICT     path to predicted images
  --num_classes NUM_CLASSES
                        number of classes for confusion matrix
  --results RESULTS     path to saving results directory
  --norm_conf           plot normalize confusion matrix
  --figure              plot the result with scores
  --write_csv           whether save csv or not
```
