#! /usr/bin/env python
import os
import time
import shutil
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pdb

import torch.nn.functional as F
from utils import import_class, AverageMeter, accuracy, LearningRate
import processors

class DML(processors.Processor):
    def load_model(self):
        r""" Initialize teacher and student """
        self.teacher = self._load_one_model(self.arg.teacher,
                                            self.arg.teacher_model_args)
        self.student = self._load_one_model(self.arg.student,
                                            self.arg.student_model_args)

    def load_loss(self):
        self.loss = self._load_one_loss(self.arg.loss, self.arg.loss_args)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizerT = self._load_one_sgd(self.teacher)
            self.optimizerS = self._load_one_sgd(self.student)
        elif self.arg.optimizer == 'Adam':
            self.optimizerT = optim.Adam(self.teacher.parameters())
            self.optimizerS = optim.Adam(self.student.parameters())
        else:
            raise ValueError()
        self.lr_scheduler = LearningRate([self.optimizerT, self.optimizerS],
                                         **self.arg.lr_decay)

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.global_step = self.arg.start_epoch = state['epoch']
        self.teacher.module.load_state_dict(state['teacher'],
                                            strict=self.arg.strict_load)
        self.student.module.load_state_dict(state['student'],
                                            strict=self.arg.strict_load)
        if optim:
            self.optimizerT.load_state_dict(state['optimizerT'])
            self.optimizerS.load_state_dict(state['optimizerS'])
        self.print_log('Load checkpoints from {}'.format(filename))
        return self.arg.start_epoch

    def eval(self, epoch):
        self.teacher.eval()
        self.student.eval()

        losses = [AverageMeter() for _ in range(2)]
        accs = [AverageMeter() for _ in range(2)]

        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()

                # model1: teacher
                outputs, _ = self.teacher(inputs)
                loss = F.cross_entropy(outputs, targets)
                prec, = accuracy(outputs, targets, topk=(1,))
                losses[0].update(loss.item(), inputs.size(0))
                accs[0].update(prec.item(), inputs.size(0))

                # model2: student
                outputs, _ = self.student(inputs)
                loss = F.cross_entropy(outputs, targets)
                prec, = accuracy(outputs, targets, topk=(1,))
                losses[1].update(loss.item(), inputs.size(0))
                accs[1].update(prec.item(), inputs.size(0))

        # Write to tensorboard
        self.writer.add_scalar('test/accT', accs[0].avg, self.global_step)
        self.writer.add_scalar('test/lossT', losses[0].avg, self.global_step)
        self.writer.add_scalar('test/accS', accs[1].avg, self.global_step)
        self.writer.add_scalar('test/lossS', losses[1].avg, self.global_step)

        str = ('test ' +
               ' - accT: {:.2%}'.format(accs[0].avg) +
               ' - accS: {:.2%}'.format(accs[1].avg))
        mark_acc = max(accs[0].avg, accs[1].avg)
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
        self.teacher.train()
        self.student.train()

        losses = [AverageMeter() for _ in range(6)]
        accs = [AverageMeter() for _ in range(2)]

        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

            # forward & update teacher
            outputsT, _ = self.teacher(inputs)
            outputsS, _ = self.student(inputs)
            lossT_ce = F.cross_entropy(outputsT, targets)
            lossT_kl = self.loss(outputsT, outputsS)
            lossT = lossT_ce + self.arg.kl_loss_weight * lossT_kl
            precT, = accuracy(outputsT, targets, topk=(1,))
            losses[0].update(lossT_ce.item(), inputs.shape[0])
            losses[1].update(lossT_kl.item(), inputs.shape[0])
            losses[2].update(lossT.item(), inputs.shape[0])
            accs[0].update(precT.item(), inputs.shape[0])

            self.optimizerT.zero_grad()
            lossT.backward()
            self.optimizerT.step()

            # forward & update student
            outputsT, _ = self.teacher(inputs)
            outputsS, _ = self.student(inputs)
            lossS_ce = F.cross_entropy(outputsS, targets)
            lossS_kl = self.loss(outputsS, outputsT)
            lossS = lossS_ce + self.arg.kl_loss_weight * lossS_kl
            precS, = accuracy(outputsS, targets, topk=(1,))
            losses[3].update(lossS_ce.item(), inputs.shape[0])
            losses[4].update(lossS_kl.item(), inputs.shape[0])
            losses[5].update(lossS.item(), inputs.shape[0])
            accs[1].update(precS.item(), inputs.shape[0])

            self.optimizerS.zero_grad()
            lossS.backward()
            self.optimizerS.step()

        # Write to tensorboard
        self.writer.add_scalar('train/accT', accs[0].avg, self.global_step)
        self.writer.add_scalar('train/accS', accs[1].avg, self.global_step)
        self.writer.add_scalar('train/lossT_ce', losses[0].avg, self.global_step)
        self.writer.add_scalar('train/lossT_kl', losses[1].avg, self.global_step)
        self.writer.add_scalar('train/lossT', losses[2].avg, self.global_step)
        self.writer.add_scalar('train/lossS_ce', losses[3].avg, self.global_step)
        self.writer.add_scalar('train/lossS_kl', losses[4].avg, self.global_step)
        self.writer.add_scalar('train/lossS', losses[5].avg, self.global_step)
        self.print_log('train' +
                       ' - accT: {:.2%}'.format(accs[0].avg) +
                       ' - accS: {:.2%}'.format(accs[1].avg))

    def start(self):
        if self.arg.phase == 'train':
            # Print out information log
            self.print_log('{} samples in trainset'.format(
                len(self.data_loader['train'].dataset)))
            self.print_log('{} samples in testset'.format(
                len(self.data_loader['test'].dataset)))
            if self.arg.print_model:
                self.print_log('Model1 Architecture:\n{}'.format(self.teacher))
                self.print_log('Model2 Architecture:\n{}'.format(self.student))
            num_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
            self.print_log('Teacher Parameters: {}'.format(num_params))
            num_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            self.print_log('Student Parameters: {}'.format(num_params))
            if self.arg.weights:
                self.load_checkpoint(self.arg.weights, optim=False)
                self.arg.start_epoch = 0
            self.print_log('Configurations:\n{}\n'.format(str(vars(self.arg))))

            self.global_step = 0
            self.best_acc = 0
            self.best_epoch = -1
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Decay at every epoch
                lr = self.lr_scheduler.step(epoch)
                self.print_log('\nEpoch: {}/{} - LR: {:6f}'.format(
                    epoch+1, self.arg.num_epoch, lr), print_time=False)

                # Train and test
                self.train(epoch)
                is_best = self.eval(epoch)

                # Save
                is_best = is_best and (epoch + 1) >= self.arg.save_interval
                is_interval = ((epoch + 1) % self.arg.save_interval==0) or (epoch + 1 == self.arg.num_epoch)
                if is_best or is_interval:
                    self.save_checkpoint(epoch,
                         {'epoch': epoch + 1,
                          'teacher': self.teacher.module.state_dict(),
                          'student': self.student.module.state_dict(),
                          'optimizerT': self.optimizerT.state_dict(),
                          'optimizerS': self.optimizerS.state_dict(),
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
