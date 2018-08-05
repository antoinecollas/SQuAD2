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
    - weights_100_000: pretrained model on europarl (more details in results.xlsx)
    - train.py: train/evaluate the model
    - europarl folder: it contains subsets of the dataset europarl (http://www.statmt.org/europarl/)
