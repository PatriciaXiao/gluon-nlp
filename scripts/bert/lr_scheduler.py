from __future__ import print_function
import math
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
import numpy as np

class TriangularSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):     
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction
        
    def __call__(self, iteration):
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle

class CosineAnnealingSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.0):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction       
    def __call__(self, iteration):
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1. / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos((iteration - self.cycle_length*self.inc_fraction) * math.pi / (self.cycle_length * (1 - self.inc_fraction)))) / 2
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle

class OneCycleSchedule():
    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if (cooldown_length > 0) and (finish_lr is None):
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if (cooldown_length == 0) and (finish_lr is not None):
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")
            
        finish_lr = finish_lr if (cooldown_length > 0) else start_lr
        schedule = TriangularSchedule(min_lr=start_lr, max_lr=max_lr, cycle_length=cycle_length)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)
        
    def __call__(self, iteration):
        return self.schedule(iteration)

class TrapezoidSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, con_fraction=0.5, inc_fraction=0.0):     
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        con_fraction: fraction of iterations spent in holding constant lr
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction
        self.con_fraction = con_fraction     
        assert self.inc_fraction + self.con_fraction <= 1., "increading fraction and constant fraction couldn't have sum larger than 1"   
    def __call__(self, iteration):
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length*(self.inc_fraction + self.con_fraction):
            unit_cycle = 1
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction - self.con_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle

def plot_lr_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()

class LearningRateSchedule(object):
    def __init__(self, args, batch_size, num_train_examples):
        self.get_lr_params(args, batch_size, num_train_examples)
        self.initialize(args.lr_scheduler)

    def initialize(self, name):
        self.step = 0
        self.scheduler = self.get_lr_scheduler(name, **self.params)

    def update(self, num_steps=1):
        self.step += num_steps

    def __call__(self):
        return self.scheduler(self.step)

    def get_lr_params(self, args, batch_size, num_train_examples):
        self.batch_size = batch_size
        self.num_train_examples = num_train_examples
        self.step_size = self.batch_size * args.accumulate if args.accumulate else self.batch_size
        self.num_train_steps = int(self.num_train_examples / self.step_size * args.epochs)
        self.num_warmup_steps = int(self.num_train_steps * args.warmup_ratio)
        self.warmup_ratio = args.warmup_ratio
        params_list = {
            "FactorScheduler": {
                    'base_lr': args.lr,
                    'step': int(self.num_train_steps / 5), 
                    'factor': 0.8,
                    'stop_factor_lr': 1e-8,
                    'warmup_steps': self.num_warmup_steps, 
                    'warmup_begin_lr': args.min_lr, 
                    'warmup_mode':'linear'
                },
            "MultiFactorScheduler": {
                    'base_lr': args.lr,
                    'step': [self.num_train_steps / 5],
                    'factor': 0.8,
                    'warmup_steps': self.num_warmup_steps, 
                    'warmup_begin_lr': args.min_lr, 
                    'warmup_mode':'linear'
                },
            "PolyScheduler": {
                    'base_lr': args.lr,
                    'max_update': self.num_train_steps, 
                    'final_lr': args.min_lr,
                    'warmup_steps': self.num_warmup_steps,
                    'warmup_begin_lr': args.min_lr, 
                    'warmup_mode':'linear'
                },
            "TriangularSchedule": {
                    'min_lr': args.min_lr,
                    'max_lr': args.lr,
                    'cycle_length': self.num_train_steps,
                    'inc_fraction': self.warmup_ratio
                },
            "CosineAnnealingSchedule": {
                    'min_lr': args.min_lr,
                    'max_lr': args.lr,
                    'cycle_length': self.num_train_steps,
                    'inc_fraction': self.warmup_ratio
                },
            "OneCycleSchedule": {
                    'start_lr': args.min_lr,
                    'max_lr': args.lr,
                    'cycle_length': self.num_train_steps,
                    'cooldown_length': 0,
                    'finish_lr': None
                },
            "TrapezoidSchedule": {
                    'min_lr': args.min_lr,
                    'max_lr': args.lr,
                    'cycle_length': self.num_train_steps,
                    'inc_fraction': 0.0,
                    'con_fraction': self.warmup_ratio
                }
        }
        self.params = params_list[args.lr_scheduler]

    def get_lr_scheduler(self, name, **kwargs):
        scheduler_list = {
            "FactorScheduler": mx.lr_scheduler.FactorScheduler,
            "MultiFactorScheduler": mx.lr_scheduler.MultiFactorScheduler,
            "PolyScheduler": mx.lr_scheduler.PolyScheduler,
            "TriangularSchedule": TriangularSchedule,
            "CosineAnnealingSchedule": CosineAnnealingSchedule,
            "OneCycleSchedule": OneCycleSchedule,
            "TrapezoidSchedule": TrapezoidSchedule
        }
        lr_scheduler = scheduler_list[name](**kwargs)
        return lr_scheduler

