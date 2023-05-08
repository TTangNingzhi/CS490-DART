# README

This is the official implementation of the
paper [DART: Resolving Training Biases for Recommender Systems via Data Discarding and Relabeling](https://arxiv.org/abs/xxxx.yyyyy)
.

## Hierarchy

    .
    ├── torchfm # TorchFM library with some modifications
    │   ├── dataset
    │   │   ├── __init__.py
    │   │   ├── avazu.py    # Modified code for Avazu -- Setting 1
    │   │   ├── criteo.py   # Modified code for Criteo -- Setting 1
    │   │   ├── custom.py   # Code for custom dataset (all datasets except MovieLens, Avazu and Criteo) -- Setting 2
    │   │   ├── movielens_1.py  # Modified code for MovieLens -- Setting 1
    │   │   ├── movielens_2.py  # Modified code for MovieLens -- Setting 2
    │   ├── model
    │   │   ├── __init__.py
    │   │   ├── fm.py   # FM model
    │   │   ├── dfm.py  # DFM model
    │   │   ├── ...
    ├── test
    │   ├── test_layers.py  # Test for layers
    ├── examples
    │   ├── arxiv   # Arxiv deprecated code
    │   ├── chkpt   # Checkpoints
    │   ├── data    # Datasets
    │   │   ├── adressa
    │   │   ├── Jester
    │   │   ├── book_crossing
    │   │   ├── ml-100k
    │   │   ├── ml-1m
    │   │   ├── ...
    │   ├── logs    # Logs, Processing Codes and Results
    │   │   ├── [2023-05-06]
    │   │   ├── ...
    │   ├── config.py   # Configuration file to set random seed
    │   ├── main_1.py   # Main function for Setting 1
    │   ├── main_2.py   # Main function for Setting 2
    ├── README.md # This file

## Requirements

Here are the requirements to run the code:

```
python==3.9.7
torch==1.10.1+cu102
tqdm==4.62.3
numpy==1.21.5
```

## Dataset

We use the following datasets in our experiments, which are all publicly available:

+ [MovieLens](https://grouplens.org/datasets/movielens/)
+ [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)
+ [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
+ [Jester](https://goldberg.berkeley.edu/jester-data/)
+ [Book-Crossing](https://grouplens.org/datasets/book-crossing/)
+ [Adressa](https://reclab.idi.ntnu.no/dataset/)
+ [Yelp](https://www.yelp.com/dataset)
+ [Amazon-book](https://nijianmo.github.io/amazon/index.html)

The datasets should be stored in the `examples/data` folder separately. For several of them, you need to run the
preprocessing code in each folder to process the raw data into the required format.

## Setting

### Setting 1

For the first setting, we use the MovieLens, Avazu and Criteo datasets. Here we simulate false-positive interactions by
randomly sampling noise.

### Setting 2

For the second setting, we use the Jester, Book-Crossing, Adressa, Yelp, Amazon and Gowalla datasets. Here we use a more
reasonable method to model false-positive interactions as described in the paper.

## Method

We implement the following methods in our experiments:

+ Base (none)
+ DART (mixed)
+ [ADT-CE-T](https://dl.acm.org/doi/10.1145/3437963.3441800) (discard)
+ [ADT-CE-R](https://dl.acm.org/doi/10.1145/3437963.3441800) (reweight)
+ [LCD](https://dl.acm.org/doi/abs/10.1145/3511808.3557625) (lcd)
+ [LCD-Re](https://dl.acm.org/doi/abs/10.1145/3511808.3557625) (lcd-re)
+ [Bootstrap-Soft](https://arxiv.org/abs/1412.6596) (bootstrap_soft)
+ [Bootstrap-Hard](https://arxiv.org/abs/1412.6596) (bootstrap_hard)

## Usage

### Run

To run the code, you can use the following command in `/examples` folder:

```
usage: main_2.py [-h] [--dataset_name DATASET_NAME] [--model_name MODEL_NAME] [--epoch EPOCH]
                 [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                 [--weight_decay WEIGHT_DECAY] [--device DEVICE] [--save_dir SAVE_DIR]
                 [--method METHOD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        criteo, avazu, movielens100K, movielens1M, or adressa, amazon-book,
                        yelp, book-crossing, jester
  --model_name MODEL_NAME
  --epoch EPOCH
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --weight_decay WEIGHT_DECAY
  --device DEVICE
  --save_dir SAVE_DIR
  --method METHOD       none, mixed, discard, reweight, lcd, lcd-re, bootstrap_soft,
                        or bootstrap_hard
```

### Hyper-parameters

Our four hyper-parameters $T_i, T_m, \tau_m, r$ are set in the `train()` method of `main_2.py`

```
Ti, Tm, taum, r = 0, 20000, 0.04, 0
```

It is worth noting that each experimental result may be adapted to different hyper-parameters to obtain the best
results,
depending on the specific dataset. Please refer to the Sensitivity Analysis section in the paper for details.

## Citation

If you find this code useful, please cite our paper:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={},
  organization={}
}
```

## Contact

If you have any questions, please contact Ningzhi Tang via email: `tangnz2019@mail.sustech.edu.cn`.

## Acknowledgement

We thank the authors of [TorchFM](https://github.com/rixwew/pytorch-fm) for their open-source code.