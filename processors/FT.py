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

class FactorTransfer(processors.Processor):
    def load_model(self):
        r""" Initialize models """
        self.teacher = self._load_one_model(self.arg.teacher,
                                            self.arg.teacher_model_args)
        self.student = self._load_one_model(self.arg.student,
                                            self.arg.student_model_args)
        self.embedT = self._load_one_model(self.arg.embed,
                                           self.arg.teacher_embed_args)
        self.embedS = self._load_one_model(self.arg.embed,
                                           self.arg.student_embed_args)

        # load teacher
        if not self.arg.teacher_weights:
            raise ValueError('Please appoint --teacher-weights.')
        self.load_weight(self.teacher, self.arg.teacher_weights, 'teacher', 'model')

    def load_loss(self):
        r""" Construct losses and decayed weight for each stage """
        self.loss = self._load_one_loss(self.arg.loss, self.arg.loss_args)
        self.rec_loss = self._load_one_loss('RecLoss')

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizerT = self._load_one_sgd(self.embedT)
            self.optimizerS = self._load_one_sgd(self.student, self.embedS)
        elif self.arg.optimizer == 'Adam':
            raise NotImplementedError
        else:
            raise ValueError()
        self.lr_schedulerT = LearningRate(self.optimizerT,
                                          **self.arg.lr_decayT)
        self.lr_schedulerS = LearningRate(self.optimizerS,
                                          **self.arg.lr_decay)

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.global_step = self.arg.start_epoch = state['epoch']
        self.embedT.module.load_state_dict(state['embedT'],
                                           strict=self.arg.strict_load)
        self.student.module.load_state_dict(state['student'],
                                            strict=self.arg.strict_load)
        self.embedS.module.load_state_dict(state['embedS'],
                                           strict=self.arg.strict_load)
        if optim:
            self.optimizerT.load_state_dict(state['optimizerT'])
            self.optimizerS.load_state_dict(state['optimizerS'])
        self.print_log('Load checkpoints from {}'.format(filename))
        return self.arg.start_epoch

    def eval_teacher(self, epoch):
        self.teacher.eval()
        self.embedT.eval()

        losses = [AverageMeter() for _ in range(1)]
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()

                # teacher
                _, features = self.teacher(inputs)
                _, rec_features = self.embedT(features)
                loss = self.rec_loss(features, rec_features)
                losses[0].update(loss.item(), inputs.shape[0])

        self.writer.add_scalar('test/lossRec', losses[0].avg, self.global_step)

        self.print_log('test  - loss: {:.2f}'.format(losses[0].avg))

    def eval_student(self, epoch):
        self.teacher.eval()
        self.student.eval()
        self.embedT.eval()
        self.embedS.eval()

        losses = [AverageMeter() for _ in range(1)]
        accs = [AverageMeter() for _ in range(1)]
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()

                outputs, _ = self.student(inputs)
                loss = F.cross_entropy(outputs, targets)
                prec, = accuracy(outputs, targets, topk=(1,))
                losses[0].update(loss.item(), inputs.shape[0])
                accs[0].update(prec.item(), inputs.shape[0])

        self.writer.add_scalar('test/accS', accs[0].avg, self.global_step)
        self.writer.add_scalar('test/lossS', losses[0].avg, self.global_step)

        str = 'test  - acc: {:.2%}'.format(accs[0].avg)
        mark_acc = accs[0].avg
        is_best = mark_acc > self.best_acc
        if is_best:
            self.best_acc = mark_acc
            self.best_epoch = epoch + 1
            self.print_log(str + ' (*)')
        else:
            self.print_log(str)
        return is_best

    def train_teacher(self, epoch):
        self.global_step += 1
        self.teacher.eval()
        self.embedT.train()
        self.student.eval()
        self.embedS.eval()

        losses = [AverageMeter() for _ in range(1)]
        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs = inputs.float().cuda()
            targets = targets.long().cuda()

            _, features = self.teacher(inputs)
            _, rec_features = self.embedT(features)
            loss = self.rec_loss(features, rec_features)

            self.optimizerT.zero_grad()
            loss.backward()
            self.optimizerT.step()

            losses[0].update(loss.item(), inputs.shape[0])

        self.writer.add_scalar('train/lossRec', losses[0].avg, self.global_step)
        self.print_log('train - loss: {:.2f}'.format(losses[0].avg))

    def train_student(self, epoch):
        self.global_step += 1
        self.teacher.eval()
        self.embedT.eval()
        self.student.train()
        self.embedS.train()

        losses = [AverageMeter() for _ in range(3)]
        accs = [AverageMeter() for _ in range(1)]
        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs = inputs.float().cuda()
            targets = targets.long().cuda()

            # teacher forward
            _, featuresT = self.teacher(inputs)
            factorsT, _ = self.embedT(featuresT)

            # student forward
            outputsS, featuresS = self.student(inputs)
            factorsS, _ = self.embedS(featuresS)

            # loss
            loss_ce = F.cross_entropy(outputsS, targets)
            loss_ft = self.loss(factorsS, factorsT)
            loss = loss_ce + self.arg.loss_weight * loss_ft

            # backward
            self.optimizerS.zero_grad()
            loss.backward()
            self.optimizerS.step()

            # accuracy
            prec, = accuracy(outputsS, targets, topk=(1,))
            losses[0].update(loss.item(), inputs.shape[0])
            losses[1].update(loss_ce.item(), inputs.shape[0])
            losses[2].update(loss_ft.item(), inputs.shape[0])
            accs[0].update(prec.item(), inputs.shape[0])

        self.writer.add_scalar('train/accS', accs[0].avg, self.global_step)
        self.writer.add_scalar('train/lossS', losses[0].avg, self.global_step)
        self.writer.add_scalar('train/lossCE', losses[1].avg, self.global_step)
        self.writer.add_scalar('train/lossFT', losses[2].avg, self.global_step)
        self.print_log('train - acc: {:.2%}'.format(accs[0].avg))

    def start(self):
        if self.arg.phase == 'train':
            # General configurations
            self.print_log('{} samples in trainset'.format(
                len(self.data_loader['train'].dataset)))
            self.print_log('{} samples in testset'.format(
                len(self.data_loader['test'].dataset)))
            self.print_log('Configurations:\n{}\n'.format(str(vars(self.arg))))

            # teacher's embed
            self.global_step = 0
            for epoch in range(self.arg.num_epoch_embedT):
                lr = self.lr_schedulerT.step(epoch)
                self.print_log('\nEmbedT Epoch: {}/{} - LR: {:6f}'.format(
                    epoch+1, self.arg.num_epoch_embedT, lr), print_time=False)
                self.train_teacher(epoch)
                self.eval_teacher(epoch)

            self.global_step = 0
            self.best_acc = 0
            self.best_epoch = -1
            for epoch in range(self.arg.num_epoch):
                lr = self.lr_schedulerS.step(epoch)
                self.print_log('\nStudent Epoch: {}/{} - LR: {:6f}'.format(
                    epoch+1, self.arg.num_epoch, lr), print_time=False)
                self.train_student(epoch)
                is_best = self.eval_student(epoch)

                is_best = is_best and (epoch + 1) >= self.arg.save_interval
                is_interval = (((epoch + 1) % self.arg.save_interval == 0) or
                               (epoch + 1 == self.arg.num_epoch))
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
            self.arg.print_log = False
            if self.arg.student_weights is None:
                raise ValueError('Please appoint --weights.')
            self.load_weight(self.student, self.arg.student_weights, 'student')
            self.print_log('Model:   {}.'.format(self.arg.student))
            self.print_log('Weights: {}.'.format(self.arg.student_weights))
            self.eval(epoch=0)
            self.print_log('Done.\n')

        elif self.arg.phase == 'debug':
            self.arg.print_log = False
            self.train(0)
            self.eval(0)

        else:
            raise ValueError("Unknown phase: {}".format(self.arg.phase))
