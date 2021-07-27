#!/usr/bin/env python

import pdb
import torch
import yaml
import random
import argparse

import numpy as np
import processors

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Dedistilling')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/debug',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default=None,
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--processor',
        default='retrieval',
        type=str,
        help='Type of Processor')
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # general config
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--model-save-name',
        type=str,
        default='model',
        help='Checkpoint name')
    parser.add_argument(
        '--strict-load',
        type=str2bool,
        default=False,
        help="whether load model strictly")
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--print-model',
        type=str2bool,
        default=False,
        help='print model or not')

    # dataset
    parser.add_argument(
        '--feeder',
        default='feeder.cifar100',
        type=str,
        help='Class name of data feeder')
    parser.add_argument(
        '--feeder-args',
        default=dict(),
        help='the arguments of data loader')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64)
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=64)
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')

    # hyper parameters
    parser.add_argument(
        '--phases',
        type=list,
        default=['A', 'B', 'C'],
        help='pipeline of different phases')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=1,
        help='# of epochs for training')
    parser.add_argument(
        '--num-epoch-embedS',
        type=int,
        default=1,
        help='# of epochs for training')
    parser.add_argument(
        '--num-epoch-embedT',
        type=int,
        default=1,
        help='# of epochs for training')

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='model.resnet32',
        help='Class name of model')
    parser.add_argument(
        '--model-args',
        type=dict,
        default={},
        help='Args for model')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help="Dir of weights")
    parser.add_argument(
        '--teacher',
        type=str,
        default='model.WRN28_10',
        help='Class name of teacher model')
    parser.add_argument(
        '--teacher-model-args',
        type=dict,
        default={},
        help='Args for teacher model')
    parser.add_argument(
        '--teacher-weights',
        type=str,
        default=None,
        help="Dir of teacher's weights")
    parser.add_argument(
        '--student',
        type=str,
        default='model.resnet32',
        help='Class name of student model')
    parser.add_argument(
        '--student-model-args',
        type=dict,
        default={},
        help='Args for student model')
    parser.add_argument(
        '--student-weights',
        type=str,
        default=None,
        help="Dir of student's weights")
    parser.add_argument(
        '--embed',
        type=str,
        default='auto_encoder',
        help='Class name of embedding model')
    parser.add_argument(
        '--teacher-embed-args',
        type=dict,
        default={'cin': 64, 'cout': 32},
        help="Args for teacher's embedding model")
    parser.add_argument(
        '--student-embed-args',
        type=dict,
        default={'cin': 64, 'cout': 32},
        help="Args for student's embedding model")
    parser.add_argument(
        '--teacher-embedI-args',
        type=dict,
        default={'cin': 64, 'cout': 32},
        help="Args for teacher's inhering embedding model")
    parser.add_argument(
        '--teacher-embedE-args',
        type=dict,
        default={'cin': 64, 'cout': 32},
        help="Args for teacher's exploring embedding model")

    # Loss
    parser.add_argument(
        '--loss',
        type=str,
        default='',
        help='Class name of loss')
    parser.add_argument(
        '--loss-args',
        type=dict,
        default={},
        help='Args for loss')
    parser.add_argument(
        '--inh-loss',
        type=str,
        default='',
        help='Class name of inheritance loss')
    parser.add_argument(
        '--inh-loss-args',
        type=dict,
        default={},
        help='Args for inheritance loss')
    parser.add_argument(
        '--exp-loss',
        type=str,
        default='',
        help='Class name of exploration loss')
    parser.add_argument(
        '--exp-loss-args',
        type=dict,
        default={},
        help='Args for exploration loss')
    parser.add_argument(
        '--loss-weight',
        type=float,
        default=0,
        help='Weights for distillation loss')
    parser.add_argument(
        '--kl-loss-weight',
        type=float,
        default=0,
        help='Weights for kl div loss')
    parser.add_argument(
        '--kl-loss-weight-decay',
        type=dict,
        default={'base': 0},
        help='Weight decay for kl div loss')
    parser.add_argument(
        '--inh-loss-weight',
        type=float,
        default=0,
        help='Weights for inheritance loss')
    parser.add_argument(
        '--inh-loss-weight-decay',
        type=dict,
        default={'base': 0},
        help='Weight decay for inheritance loss')
    parser.add_argument(
        '--exp-loss-weight',
        type=float,
        default=0,
        help='Weights for exploration loss')
    parser.add_argument(
        '--exp-loss-weight-decay',
        type=dict,
        default={'base': 0},
        help='Weight decay for exploration loss')
    parser.add_argument(
        '--rec-loss-weight',
        type=float,
        default=0,
        help='Weight for reconstruction loss')
    parser.add_argument(
        '--rec-loss-weight-decay',
        type=dict,
        default={'base': 0},
        help='Weight decay for reconstruction loss')

    # Dedistillation
    parser.add_argument(
        '--divide',
        type=str,
        default='natural',
        help='Strategy for channel division')

    # optim
    parser.add_argument(
        '--optimizer',
        type=str,
        default='SGD',
        help='Type of optimizer')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for SGD optimizer')
    parser.add_argument(
        '--weight-decayT',
        type=float,
        default=0.0005,
        help='weight decay for teacher model')
    parser.add_argument(
        '--weight-decayS',
        type=float,
        default=0.0005,
        help='weight decay for student')
    parser.add_argument(
        '--weight-decay-embed',
        type=float,
        default=0.0005,
        help='weight decay for embedding model parameters')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=True,
        help='use nesterov or not')
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--base-lr-embed',
        type=float,
        default=0.01,
        help='initial learning rate for embedding module')
    parser.add_argument(
        '--lr-decay',
        type=dict,
        default={'base': 0.1,
                 'milestones': [60, 120, 180],
                 'gammas': 0.1},
        help='Learning rate args')
    parser.add_argument(
        '--lr-decayT',
        type=dict,
        default={},
        help='Learning rate args for teacher')
    parser.add_argument(
        '--lr-decay-embedT',
        type=dict,
        default={},
        help="Learning rate args for teacher's embedding module")
    parser.add_argument(
        '--lr-decayS',
        type=dict,
        default={},
        help='Learning rate args for student')
    parser.add_argument(
        '--lr-decay-embedS',
        type=dict,
        default={},
        help="Learning rate args for student's embedding module")

    # testing
    parser.add_argument(
        '--top5',
        type=str2bool,
        default=False)

    # multi-gpu
    parser.add_argument(
        '--local_rank',
        type=int)
    parser.add_argument(
        '--mgpu',
        type=str2bool,
        default=False)
    return parser


if __name__ == '__main__':
    # config, unparsed = get_config()
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"Wrong Arg: {k}")
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    Processor = getattr(processors, arg.processor)
    p = Processor(arg)
    p.start()
