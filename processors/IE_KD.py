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
from utils import import_class, AverageMeter, accuracy, LearningRate, LossWeightDecay
import processors


class IE_KD(processors.Processor):
    def load_model(self):
        r""" Initialize models """
        self.teacher = self._load_one_model(self.arg.teacher,
                                            self.arg.teacher_model_args)
        self.student = self._load_one_model(self.arg.student,
                                            self.arg.student_model_args)

        self.embedTI = self._load_one_model(self.arg.embed,
                                            self.arg.teacher_embedI_args)
        self.embedTE = self._load_one_model(self.arg.embed,
                                            self.arg.teacher_embedE_args)
        self.embedS = self._load_one_model(self.arg.embed,
                                           self.arg.student_embed_args)

        # load student
        if not self.arg.student_weights:
            raise ValueError('Please appoint --student-weights.')
        self.load_weight(self.student, self.arg.student_weights, 'student',
                         'model')

    def load_loss(self):
        self.inh_loss = self._load_one_loss(self.arg.inh_loss,
                                            self.arg.inh_loss_args)
        self.exp_loss = self._load_one_loss(self.arg.exp_loss,
                                            self.arg.exp_loss_args)
        self.rec_loss = self._load_one_loss('RecLoss')
        self.inh_loss_weight = LossWeightDecay(
            **self.arg.inh_loss_weight_decay)
        self.exp_loss_weight = LossWeightDecay(
            **self.arg.exp_loss_weight_decay)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizerT = self._load_one_sgd(self.teacher, self.embedTI,
                                                 self.embedTE)
            self.optimizerS = self._load_one_sgd(self.embedS)
            self.optimizer_embedT = self._load_one_sgd(self.embedTI,
                                                       self.embedTE)
        elif self.arg.optimizer == 'Adam':
            raise NotImplementedError
        else:
            raise ValueError()
        self.lr_schedulerT = LearningRate(self.optimizerT,
                                          **self.arg.lr_decayT)
        self.lr_scheduler_embedT = LearningRate(self.optimizer_embedT,
                                                **self.arg.lr_decay_embedT)
        self.lr_schedulerS = LearningRate(self.optimizerS,
                                          **self.arg.lr_decayS)

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.global_step = self.arg.start_epoch = state['epoch']
        self.teacher.module.load_state_dict(state['teacher'],
                                            strict=self.arg.strict_load)
        self.student.module.load_state_dict(state['student'],
                                            strict=self.arg.strict_load)
        self.embedTI.module.load_state_dict(state['embedTI'],
                                            strict=self.arg.strict_load)
        self.embedTE.module.load_state_dict(state['embedTE'],
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
        self.embedTI.eval()
        self.embedTE.eval()
        self.student.eval()
        self.embedS.eval()

        losses = AverageMeter()
        accs = AverageMeter()
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()

                # teacher
                outputs, _ = self.teacher(inputs)
                loss = F.cross_entropy(outputs, targets)
                prec, = accuracy(outputs, targets, topk=(1, ))
                losses.update(loss.item(), inputs.size(0))
                accs.update(prec.item(), inputs.size(0))

        self.writer.add_scalar('test/accT', accs.avg, self.global_step)
        self.writer.add_scalar('test/lossT', losses.avg, self.global_step)

        is_best = accs.avg > self.best_acc
        str = ('test ' + ' - acc: {:.2%}'.format(accs.avg) +
               ' - loss: {:.2f}'.format(losses.avg))
        if is_best:
            self.best_acc = accs.avg
            self.best_epoch = epoch + 1
            self.print_log(str + ' (*)')
        else:
            self.print_log(str)
        return is_best

    def eval_student(self, epoch):
        self.teacher.eval()
        self.student.eval()
        self.embedTI.eval()
        self.embedTE.eval()
        self.embedS.eval()

        losses = AverageMeter()
        accs = AverageMeter()
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()

                outputs, features = self.student(inputs)
                _, rec_features = self.embedS(features)

                loss = self.rec_loss(features, rec_features)
                prec, = accuracy(outputs, targets, topk=(1, ))
                losses.update(loss.item(), inputs.size(0))
                accs.update(prec.item(), inputs.size(0))

        self.writer.add_scalar('test/accS', accs.avg, self.global_step)
        self.writer.add_scalar('test/lossRecS', losses.avg, self.global_step)

        self.print_log('test ' + ' - lossRecS: {:.2f}'.format(losses.avg) +
                       ' - acc: {:.2%}'.format(accs.avg))

    def train_student(self, epoch):
        self.global_step += 1
        self.teacher.eval()
        self.embedTI.eval()
        self.embedTE.eval()
        self.student.eval()
        self.embedS.train()

        losses = AverageMeter()
        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

            _, features = self.student(inputs)
            _, rec_features = self.embedS(features)
            loss = self.rec_loss(features, rec_features)

            self.optimizerS.zero_grad()
            loss.backward()
            self.optimizerS.step()

            losses.update(loss.item(), inputs.size(0))

        self.writer.add_scalar('train/lossRecS', losses.avg, self.global_step)
        self.print_log('train' + ' - lossRecS: {:.2f}'.format(losses.avg))

    def train_teacher(self, epoch, inh_loss_weight, exp_loss_weight):
        self.global_step += 1
        self.teacher.train()
        self.embedTI.train()
        self.embedTE.train()
        self.student.eval()
        self.embedS.eval()

        losses = [AverageMeter() for _ in range(4)]
        accs = AverageMeter()
        process = tqdm(self.data_loader['train'])
        for batch_idx, (inputs, targets) in enumerate(process):
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

            # teacher forward
            outputsT, featuresT = self.teacher(inputs)
            featuresTI, featuresTE = self.divide_features(featuresT)
            factorsTI, _ = self.embedTI(featuresTI)
            factorsTE, _ = self.embedTE(featuresTE)

            # student forward
            _, featuresS = self.student(inputs)
            factorsS, _ = self.embedS(featuresS)

            # loss
            loss_ce = F.cross_entropy(outputsT, targets)
            loss_inh = self.inh_loss(factorsS, factorsTI)
            loss_exp = self.exp_loss(factorsS, factorsTE)
            loss = loss_ce + inh_loss_weight * loss_inh + exp_loss_weight * loss_exp

            # backward
            self.optimizerT.zero_grad()
            loss.backward()
            self.optimizerT.step()

            # accuracy
            prec = accuracy(outputsT, targets, topk=(1, ))[0]
            losses[0].update(loss.item(), inputs.size(0))
            losses[1].update(loss_ce.item(), inputs.size(0))
            losses[2].update(loss_inh.item(), inputs.size(0))
            losses[3].update(loss_exp.item(), inputs.size(0))
            accs.update(prec.item(), inputs.size(0))

        self.writer.add_scalar('train/accT', accs.avg, self.global_step)
        self.writer.add_scalar('train/lossT', losses[0].avg, self.global_step)
        self.writer.add_scalar('train/lossCE', losses[1].avg, self.global_step)
        self.writer.add_scalar('train/lossInh', losses[2].avg,
                               self.global_step)
        self.writer.add_scalar('train/lossExp', losses[3].avg,
                               self.global_step)
        self.print_log('train' + ' - acc: {:.2%}'.format(accs.avg) +
                       ' - loss: {:.2f}'.format(losses[0].avg) +
                       ' - CE: {:.2f}'.format(losses[1].avg) +
                       ' - Inh: {:.2f}'.format(losses[2].avg) +
                       ' - Exp: {:.2f}'.format(losses[3].avg))

    def start(self):
        if self.arg.phase == 'train':
            # General configurations
            self.print_log('{} samples in trainset'.format(
                len(self.data_loader['train'].dataset)))
            self.print_log('{} samples in testset'.format(
                len(self.data_loader['test'].dataset)))
            self.print_log('Configurations:\n{}\n'.format(str(vars(self.arg))))

            # train student's embed
            for epoch in range(0, self.arg.num_epoch_embedS):
                lr = self.lr_schedulerS.step(epoch)
                self.print_log('\nEmbedS Epoch: {}/{} - LR: {}'.format(
                    epoch + 1, self.arg.num_epoch_embedS, lr),
                               print_time=False)
                self.train_student(epoch)
                self.eval_student(epoch)

            self.global_step = 0
            self.best_acc = 0
            self.best_epoch = -1

            self.global_step = 0
            self.best_acc = 0
            self.best_epoch = -1
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Decay at every epoch
                lr = self.lr_schedulerT.step(epoch)
                inh_loss_weight = self.inh_loss_weight.step(epoch)
                exp_loss_weight = self.exp_loss_weight.step(epoch)
                self.print_log('\nTeacher Epoch: {}/{} - LR: {:.8f}'
                               ' - Inh: {} - Exp: {}'.format(
                                   epoch + 1, self.arg.num_epoch, lr,
                                   inh_loss_weight, exp_loss_weight),
                               print_time=False)

                # Train and test
                self.train_teacher(epoch, inh_loss_weight, exp_loss_weight)
                is_best = self.eval_teacher(epoch)

                # Save
                is_best = is_best and (epoch + 1) >= self.arg.save_interval
                is_interval = ((epoch + 1) % self.arg.save_interval
                               == 0) or (epoch + 1 == self.arg.num_epoch)
                if is_best or is_interval:
                    self.save_checkpoint(
                        epoch, {
                            'epoch': epoch + 1,
                            'teacher': self.teacher.module.state_dict(),
                            'student': self.student.module.state_dict(),
                            'embedTI': self.embedTI.module.state_dict(),
                            'embedTE': self.embedTE.module.state_dict(),
                            'embedS': self.embedS.module.state_dict(),
                            'optimizerT': self.optimizerT.state_dict(),
                            'optimizerS': self.optimizerS.state_dict(),
                            'inh_index': self.inh_index,
                            'exp_index': self.exp_index,
                            'best_acc': self.best_acc
                        }, is_best)

            self.print_log(
                '\nBest accuracy: {:.2%}, Epoch: {}, Dir: {}, Time: {:.0f}'.
                format(self.best_acc, self.best_epoch, self.work_dir,
                       time.time() - self.start_time))

        elif self.arg.phase == 'test':
            self.arg.print_log = False
            if self.arg.teacher_weights is None:
                raise ValueError('Please appoint --teacher-weights.')
            self.load_weight(self.teacher, self.arg.teacher_weights, 'model')
            self.print_log('Model:   {}.'.format(self.arg.teacher))
            self.print_log('Weights: {}.'.format(self.arg.teacher_weights))
            self.eval(epoch=0)
            self.print_log('Done.\n')

        elif self.arg.phase == 'debug':
            self.arg.print_log = False
            self.train(0)
            self.eval(0)

        else:
            raise ValueError("Unknown phase: {}".format(self.arg.phase))

    def divide_features(self, feature):
        inh_index = getattr(self, 'inh_index', None)
        exp_index = getattr(self, 'exp_index', None)
        if inh_index is None or exp_index is None:
            self.print_log('Init Inh and Exp Index')
            length = feature.size(1)
            if self.arg.divide == 'random':
                index = torch.randperm(length).cuda()
                self.inh_index = index[:int(length / 2)]
                self.exp_index = index[int(length / 2):]
            elif self.arg.divide == 'natural':
                index = torch.arange(length).cuda()
                self.inh_index = index[:int(length / 2)]
                self.exp_index = index[int(length / 2):]
            elif self.arg.divide == 'relation':
                raise NotImplementedError
            else:
                raise ValueError()
            self.print_log('Inh Index: {}'.format(self.inh_index.tolist()))
            self.print_log('Exp Index: {}'.format(self.exp_index.tolist()))
            print()
        inh_feature = feature.index_select(dim=1, index=self.inh_index)
        exp_feature = feature.index_select(dim=1, index=self.exp_index)
        return inh_feature, exp_feature
