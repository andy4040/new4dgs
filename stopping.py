"""Plateau decisions on comparable, deterministic evaluation losses."""
import math


class Plateau:
    def __init__(self, patience=5, min_delta=1e-4, min_steps=2000):
        if patience < 1 or min_delta < 0 or min_steps < 0:
            raise ValueError('Invalid stopping parameters')
        self.patience=patience;self.min_delta=min_delta;self.min_steps=min_steps
        self.best=math.inf;self.anchor=math.inf;self.best_step=0;self.bad=0

    def update(self, value, step):
        if not math.isfinite(value):raise FloatingPointError('Nonfinite monitored loss')
        improved=value < self.best
        if improved:self.best=value;self.best_step=step
        if value < self.anchor-self.min_delta:
            self.anchor=value;self.bad=0
        elif step >= self.min_steps:self.bad+=1
        return improved, step >= self.min_steps and self.bad >= self.patience

    def reset_patience(self):
        self.anchor=math.inf;self.bad=0

    def report(self):
        return dict(best_loss=self.best,best_step=self.best_step,bad_evaluations=self.bad,
                    patience=self.patience,min_delta=self.min_delta,min_steps=self.min_steps)


def stopping_config(cfg):
    return {'enabled':True,'eval_interval':1000,'patience':5,'min_delta':1e-4,
            'min_steps':2000,**cfg.get('early_stopping',{})}


def make_stopper(cfg):
    c=stopping_config(cfg)
    return Plateau(c['patience'],c['min_delta'],c['min_steps'])
