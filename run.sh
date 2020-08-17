#!/bin/bash

python3 main.py --gpu 0 --dataset modelnet40 --poseloss 120 --log-dir ./tests ./data
