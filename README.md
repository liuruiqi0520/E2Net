# E<sup>2</sup>Net: Resource-Efficient Continual Learning with Elastic Expansion Network

## Introduction
This is the training and evaluation code for our work "E<sup>2</sup>Net: Resource-Efficient Continual Learning with Elastic Expansion Network".


## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

 Examples:

    python main.py --dataset seq-cifar10 --model e2n --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-tinyimg --model e2n --buffer_size 500 --load_best_args
   
    python main.py --dataset seq-cifar100 --model e2n --buffer_size 500 --load_best_args
    
 

## Requirements

- torch==1.12.1

- torchvision==0.13.1

- tensorflow 2.11.0
