# Transformer

The repository is an implementation of Transformer (neural machine translation, for more info: https://arxiv.org/abs/1706.03762).

## Install environment
```
git clone https://github.com/antoinecollas/transformer_neural_machine_translation
cd transformer_neural_machine_translation
conda env update
conda activate transformer
```

## Organization
This repository is organized in two parts:
- model: contains the building blocks of the model (feed forward, self attention, ...)
- train: contains files to train the model
    - prepare_data.py: it prepares europarl data:
        - split into training and test sets, create subword units ...
    - constants.py: all the hyperparameters and files paths
    - train.py: train/evaluate the model

python setup.py install && python train/prepare_data.py --dev -s datasets/baseline-1M-enfr/baseline-1M.en -t datasets/baseline-1M-enfr/baseline-1M.fr