# CV Project Assignment

This project has two subtasks: category classification and pairwise compatibility classification.
The first task requires construcing <img, class_label> pair for classification.
The second task requires constructing <(x1,x2), binary_label> pair for classification.

## Code structures
1. data preparation is defined in data.py
2. CNN model in defined in model.py
3. training is in train_*.py
4. utils.py provides hyper-parameters

## Dependencies
1. pytorch/tensorflow>=2.0
2. tqdm
3. sklearn
4. torchvision
5. sklearn

Dataset:
- Polyvore Outfits: https://drive.google.com/open?id=1ZCDRRh4wrYDq0O3FOptSlBpQFoe0zHTw

