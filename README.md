# BPnPNet
Solving the Blind Perspective-n-Point Problem End-To-End With Robust Differentiable Geometric Optimization
Dylan Campbell, Liu Liu, and Stephen Gould
ECCV 2020

This repository is a reference implementation for [this paper](https://arxiv.org/abs/2007.14628). If you use this code in your research, please cite as:
```
@inproceedings{campbell2020solving,
  author = {Campbell, Dylan and Liu, Liu and Gould, Stephen},
  title = {Solving the Blind Perspective-n-Point Problem End-To-End With Robust Differentiable Geometric Optimization},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={preprint},
  location={Glasgow, UK},
  month = {Aug},
  year = {2020},
  organization={Springer},
}
```

This work uses the Deep Declarative Networks (DDN) framework. For the full library and other applications, use [this link](https://github.com/anucvml/ddn).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Datasets

The pre-processed (randomised) data is available for download at [this link](https://drive.google.com/file/d/1y4cbbcVEJFfB3y171GiFSZi6AtgW2nL_/view?usp=sharing) (2.1GB). Data from several datasets are included:
- [ModelNet40](https://modelnet.cs.princeton.edu/)
- [MegaDepth](https://research.cs.cornell.edu/megadepth/)
- [Data61/2D3D](https://doi.org/10.4225/08/596c56d65cded)

Please follow these links if you want to download the original datasets, and note the license information of each.

## Training

To train a model, run `main.py` with the desired dataset (modelnet40 or megadepth) and the log and data directories:

```bash
python main.py --dataset modelnet40 --poseloss <pose loss start epoch> --gpu <gpu ID> --log-dir <log-dir> <data-folder>
```

For example, the following command can be used to train BPnPNet from scratch with the pose loss introduced at epoch 120, on the ModelNet40 dataset:

```bash
python main.py --dataset modelnet40 --poseloss 120 --gpu 0 --log-dir ./logs ./data
```

## Testing

To test a model, run `main.py` from a checkpoint using the evaluate flag `-e`, with the desired dataset (modelnet40 or megadepth) and the log and data directories:

```bash
python main.py -e --resume <path to checkpoint> --dataset modelnet40 --poseloss <pose loss start epoch> --gpu <gpu ID> --log-dir <log-dir> <data-folder>
```

## Usage

```
usage: main.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--seed SEED] [--gpu GPU]
               [--num_points_train NTRAIN]
               [--log-dir LOG_DIR]
               [--dataset DATASET_NAME]
               [--poseloss POSELOSS_START_EPOCH]
               DIR

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 16)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-3)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --num_points_train    number of points for each training point-set (default: 1000)
  --log-dir LOG_DIR     directory for logging loss and accuracy
  --dataset             dataset name: modelnet40 | megadepth | data61_2d3d
  --poseloss            specify epoch at which to introduce pose loss (default: 0)
```



## License

The `bpnpnet` code is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.

## Links

- [Learning to Find Good Correspondences](https://github.com/vcg-uvic/learned-correspondence-release): reference implementation for K. Yi, E. Trulls, Y. Ono, V. Lepetit, M. Salzmann, and P. Fua, "Learning to Find Good Correspondences", CVPR 2018
- [ModelNet40](https://modelnet.cs.princeton.edu/)
- [MegaDepth](https://research.cs.cornell.edu/megadepth/)
- [Data61/2D3D](https://doi.org/10.4225/08/596c56d65cded)
