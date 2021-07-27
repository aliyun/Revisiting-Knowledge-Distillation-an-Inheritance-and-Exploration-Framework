#! /usr/bin/env python
import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from utils import import_class, LearningRate, AverageMeter, accuracy
# mgpu
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Processor():
    """ Processor for Classification """
    def __init__(self, arg):
        self.arg = arg
        self.work_dir = self.arg.work_dir
        self.writer = SummaryWriter(arg.work_dir)
        if arg.mgpu:
            dist_backend = 'nccl'
            torch.cuda.set_device(arg.local_rank)
            dist.init_process_group(backend=dist_backend)

        self.load_data()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.global_step = 0
        self.resume()
        self.start_time = time.time()

    def resume(self):
        ckpt_dir = os.path.join(self.work_dir, 'ckpt')
        if os.path.exists(ckpt_dir):
            ckpts = os.listdir(ckpt_dir)
            ckpt_dict = {}
            max_epoch = -1
            resume_file = None
            for ckpt in ckpts:
                epoch = float(ckpt.split('-')[1])
                if epoch > max_epoch:
                    max_epoch = epoch
                    resume_file = ckpt
            if resume_file is not None:
                resume_file = os.path.join(ckpt_dir, resume_file)
                if os.path.exists(resume_file):
                    print('checkpoit: {} exists'.format(resume_file))
                    answer = input('resume from it? y/n:')
                    if answer == 'y':
                        self.load_checkpoint(resume_file)
                        print('resume from {}'.format(resume_file))
                    else:
                        print('Not resume')

    def load_data(self):
        Feeder = '.'.join(['feeder', self.arg.feeder])
        Feeder = import_class(Feeder)
        self.data_loader = dict()
        train_feeder, test_feeder = Feeder(**self.arg.feeder_args)
        if self.arg.mgpu:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

            train_sampler = DistributedSampler(
                train_feeder,
                num_replicas=num_replicas,
                rank=rank)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                train_feeder,
                batch_size=int(self.arg.batch_size/num_replicas),
                sampler=train_sampler,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)
        else:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_feeder,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_feeder,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker)

    def _load_one_model(self, model, args):
        Model = '.'.join(['model', model])
        Model = import_class(Model)
        model = Model(**args)
        if self.arg.mgpu:
            return DistributedDataParallel(model.cuda(),
                                           device_ids=[self.arg.local_rank],
                                           find_unused_parameters=True)
        else:
            return nn.DataParallel(model).cuda()

    def load_model(self):
        self.model = self._load_one_model(self.arg.model, self.arg.model_args)

    def _load_one_loss(self, loss, args={}):
        Loss = '.'.join(['loss', loss])
        Loss = import_class(Loss)
        return Loss(**args).cuda()

    def load_loss(self):
        pass

    def _load_one_sgd(self, *models, nesterov=None, weight_decay=None):
        if nesterov is None:
            nesterov = self.arg.nesterov
        if weight_decay is None:
            weight_decay = self.arg.weight_decay
        return optim.SGD([{'params': m.parameters()} for m in models],
                         lr=0.1,
                         momentum=0.9,
                         nesterov=nesterov,
                         weight_decay=weight_decay)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = self._load_one_sgd(self.model)
            self.lr_scheduler = LearningRate(self.optimizer, **self.arg.lr_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise ValueError()

    def load_weight(self, model, weights, *keys):
        states = torch.load(weights)
        if not keys:
            keys = ['model']
        for k in keys:
            if k in states:
                model.module.load_state_dict(states[k])
                return
        raise ValueError('There no keys like {} in {}'.format(keys, weights))

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        if (not self.arg.mgpu) or self.arg.local_rank == 0:
            print(str)
            if self.arg.print_log:
                if not os.path.exists(self.arg.work_dir):
                    os.makedirs(self.arg.work_dir)
                with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                    print(str, file=f)

    def eval(self, epoch):
        self.model.eval()

        losses = AverageMeter()
        accs = [AverageMeter() for _ in range(2)]
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()
                outputs = self.model(inputs)

                loss = F.cross_entropy(outputs[0], targets)
                prec1, prec5 = accuracy(outputs[0], targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                accs[0].update(prec1.item(), inputs.size(0))
                accs[1].update(prec5.item(), inputs.size(0))
        self.writer.add_scalar('test/acc', accs[0].avg, self.global_step)
        self.writer.add_scalar('test/loss', losses.avg, self.global_step)

        str = 'test  - acc: {:.2%}'.format(accs[0].avg)
        if self.arg.top5:
            self.writer.add_scalar('test/acc5', accs[1].avg, self.global_step)
            str = str + ' - acc5: {:.2%}'.format(accs[1].avg)
        mark_acc = accs[0].avg
        is_best = mark_acc > self.best_acc
        if is_best:
            self.best_acc = mark_acc
            self.best_epoch = epoch + 1
            self.print_log(str + ' (*)')
        else:
            self.print_log(str)
        return is_best

    def train(self, epoch):
        self.global_step += 1
        self.model.train()

        losses = AverageMeter()
        accs = AverageMeter()
        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
            outputs, _ = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            accs.update(prec.item(), inputs.size(0))

        self.writer.add_scalar('train/acc', accs.avg, self.global_step)
        self.writer.add_scalar('train/loss', losses.avg, self.global_step)
        self.print_log('train - acc: {:.2%}'.format(accs.avg))

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.global_step = self.arg.start_epoch = state['epoch']
        self.model.module.load_state_dict(state['model'],
                                          strict=self.arg.strict_load)
        if optim:
            self.optimizer.load_state_dict(state['optimizer'])
            self.print_log('Load weights and optim from {}'.format(filename))
        self.print_log('Load weights from {}'.format(filename))
        return self.arg.start_epoch

    def save_checkpoint(self, epoch, state, is_best, save_name='model'):
        if (not self.arg.mgpu) or (self.arg.local_rank == 0):
            ckpt_dir = os.path.join(self.work_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            filename = str(epoch + 1)
            filename = save_name + '-' + filename if save_name else filename
            # ckpt_path = os.path.join(self.work_dir, 'ckpt.pth.tar')
            ckpt_path = os.path.join(ckpt_dir, filename + '-ckpt.pth.tar')
            torch.save(state, ckpt_path)
            if is_best:
                best_path = os.path.join(self.work_dir, save_name+'.pth.tar')
                torch.save(state, best_path)

    def start(self):
        if self.arg.phase == 'train':
            # Output configuration
            self.print_log('{} samples in trainset'.format(
                len(self.data_loader['train'].dataset)))
            self.print_log('{} samples in testset'.format(
                len(self.data_loader['test'].dataset)))
            if self.arg.print_model:
                self.print_log('Architecture:\n{}'.format(self.model))
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log('Parameters: {}'.format(num_params))
            if self.arg.weights:
                self.load_checkpoint(self.arg.weights, optim=False)
                self.arg.start_epoch = 0
            self.print_log('Configurations:\n{}\n'.format(str(vars(self.arg))))

            self.best_acc = 0
            self.best_epoch = -1
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Decay at every epoch
                if self.arg.optimizer == 'SGD':
                    lr = self.lr_scheduler.step(epoch)
                    self.print_log('\nEpoch: {}/{} - LR: {:6f}'.format(
                        epoch+1, self.arg.num_epoch, lr), print_time=False)
                elif self.arg.optimizer == 'Adam':
                    self.print_log('\nEpoch: {}/{}'.format(
                        epoch+1, self.arg.num_epoch), print_time=False)

                # Train and test
                self.train(epoch)
                is_best = self.eval(epoch)

                # Save
                is_best = is_best and (epoch + 1) >= self.arg.save_interval
                is_interval = ((epoch + 1) % self.arg.save_interval==0) or (epoch + 1 == self.arg.num_epoch)
                if is_best or is_interval:
                    self.save_checkpoint(epoch,
                         {'epoch': epoch + 1,
                          'model': self.model.module.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'best_acc': self.best_acc},
                         is_best)

            self.print_log('\nBest accuracy: {:.2%}, epoch: {}, dir: {}, time: {:.0f}'.format(
                self.best_acc, self.best_epoch, self.work_dir, time.time()-self.start_time))

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0)
            self.print_log('Done.\n')

        elif self.arg.phase == 'debug':
            self.arg.print_log = False
            self.train(0)
            self.eval(0)

        else:
            raise ValueError("Unknown phase: {}".format(self.arg.phase))
