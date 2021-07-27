#! /usr/bin/env python
from bisect import bisect_right
from torch.optim import Optimizer
import numpy as np

class LR(object):
    def __init__(self, type="Step", optimizer=None, **kwargs):
        self.type = type
        if type == 'Step':
            self.LR = StepLR(**kwargs)
        elif type == 'MultiStep':
            self.LR = MultiStepLR(**kwargs)
        elif type == 'MultiStepAbs':
            self.LR = MultiStepAbs(**kwargs)
        else:
            raise ValueError()

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

    def __getattr__(self, name):
        return getattr(self.LR, name)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.lr = self.LR.get()
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

class Step(object):
    def __init__(self, init=0.1, stepsize=None, gamma=0.1, last_epoch=-1):
        self.init = init
        self.stepsize = stepsize
        self.gamma = gamma
        self.last_epoch = last_epoch

    def get(self):
        if self.stepsize is None or self.stepsize <= 0:
            return self.init
        return self.gamma ** (self.last_epoch // self.stepsize)

class MultiStep(object):
    def __init__(self, init=0.1, milestones=[], gammas=0.1, last_epoch=-1):
        self.init = init
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas
        self.last_epoch = last_epoch

    def get(self):
        section = bisect_right(self.milestones, self.last_epoch)
        return self.init * np.prod(self.gammas[:section])

class MultiStepAbs(object):
    def __init__(self, init=0.1, milestones=[], gammas=0.1, last_epoch=-1):
        self.init = init
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas
        self.last_epoch = last_epoch

    def get(self):
        section = bisect_right(self.milestones, self.last_epoch)
        periods = [self.init] + self.gammas
        return periods[section]

class Coef(object):
    def __init__(self, init=0.1, milestones=[], gammas=0.1, decay='step',
                 last_epoch=-1):
        super().__init__()

        self.init = self._val = init
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas]
        if len(gammas) == 1:
            gammas = gammas * len(milestones)
        self.gammas = gammas
        self.decay = decay
        self.last_epoch = last_epoch

    @property
    def val(self):
        return self._val

    def __item__(self, i):
        return self.step(i)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if len(self.milestones) == 0 or self.decay == 'constant':
            self._val = self.init

        # step decay
        elif self.decay == "step":
            section = bisect_right(self.milestones, epoch)
            f = self.init
            for s in range(section):
                f = f * self.gammas[s]
            self._val = f

        elif self.decay == "step_abs":
            section = bisect_right(self.milestones, epoch)
            periods = [self.init] + self.gammas
            self._val = periods[section]

        else:
            raise ValueError("Unknown decay method: ", self.decay)

        return self._val


def test():
    a = Step(init=0.1)
    for i in [-1, 0, 1]:
        a.last_epoch = i
        print(a.get())
    b = Step(init=0.1, stepsize=10)
    for i in [-1, 0, 5, 10, 15]:
        b.last_epoch = i
        print(b.get())
    c = MultiStep(init=0.1)
    for i in [-1, 0, 1]:
        c.last_epoch = i
        print(c.get())
    d = MultiStep(init=0.1, milestones=[10, 20], gammas=[0.3, 0.7])
    for i in [-1, 0, 5, 10, 15, 20, 25]:
        d.last_epoch = i
        print(d.get())
    e = MultiStepAbs(init=0.1)
    for i in [-1, 0, 1]:
        e.last_epoch = i
        print(e.get())
    f = MultiStepAbs(init=0.1, milestones=[10, 20], gammas=[0.3, 0.7])
    for i in [-1, 0, 5, 10, 15, 20, 25]:
        f.last_epoch = i
        print(f.get())

    # c = Coefficient(init=0.1, milestones=[10, 20], gammas=[0.03, 0.07])
    # for i in [-1, 0, 5, 10, 15, 20, 25]:
        # print(c.step(i))
    # c = Coefficient(init=0.1, milestones=[10, 20], gammas=[0.03, 0.07],
                    # decay='step_abs')
    # for i in [-1, 0, 5, 10, 15, 20, 25]:
        # print(c.step(i))

if __name__ == "__main__":
    test()
