#coding:utf-8
import os
from tkinter import N
import torch
import argparse
import pypose as pp
from torch import nn
import matplotlib.pyplot as plt
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
import torch, warnings
from torch import nn, finfo
from pypose.optim.functional import modjac
from pypose.optim.strategy import TrustRegion
from torch.optim import Optimizer
from pypose.optim.solver import PINV, Cholesky
from torch.linalg import cholesky_ex
from pgo_dataset import G2OPGO
import numpy as np


#the parameter needs to be adjust is the alpha\thre in dis_LM, neighbor_grad_weight in rob.optim_share1  last_grad_weight in gd_armoji_share
class TrustRegion_1(object):
    def __init__(self, radius=1e6, high=.5, low=1e-3, up=2., down=.5, factor=.5, min=1e-6, max=1e16):
        super().__init__()
        assert radius > 0, ValueError("trust region radius has to be positive: {}".format(radius))
        assert high > 0, ValueError("high has to be positive: {}".format(high))
        assert low > 0, ValueError("low for decrease has to be positive: {}".format(low))
        assert 0 < down < 1, ValueError("down factor has to be smaller than 1: {}".format(down))
        assert 1 < up, ValueError("up factor has to be larger than 1: {}".format(up))
        assert 0 < factor < 1, ValueError("factor has to be smaller than 1: {}".format(factor))
        self.min, self.max, damping, self.down = min, max, 1 / radius, down
        self.defaults = {'radius':radius, 'damping':damping, 'high':high, 'low':low,
                         'up':up, 'down': down, 'factor':factor}

    def update(self, pg, last, loss, J_f, J_phi,R_f,R_phi,D,*args, **kwargs):
        quality = (last - loss) / -(J_f@D).mT @ (2*R_f-J_f@D).squeeze() -(J_phi@D).mT @(2*R_phi)
        pg['radius'] = 1. / pg['damping']
        if quality > pg['high']:
            pg['radius'] = pg['up'] * pg['radius']
            pg['down'] = self.down
        elif quality > pg['low']:
            pg['radius'] = pg['radius']
            pg['down'] = self.down
        else:
            pg['radius'] = pg['radius'] * pg['down']
            pg['down'] = pg['down'] * pg['factor']
        pg['down'] = max(self.min, min(pg['down'], self.max))
        pg['radius'] = max(self.min, min(pg['radius'], self.max))
        pg['damping'] = 1. / pg['radius']


class RobustModel_consensus(nn.Module):
    '''
    Standardize a model for least square problems with an option of square-rooting kernel.
    Then model regression becomes minimizing the output of the standardized model.
    This class is used during optimization but is not designed to expose to PyPose users.
    '''
    def __init__(self, model, kernel=None, auto=False):
        super().__init__()
        self.model = model
        self.kernel = pp.optim.optimizer.Trivial() if kernel is None else kernel

        if auto:
            self.register_forward_hook(self.kernel_forward)

    def forward(self, input, target):
        output = self.model_forward(input)
        return self.residual(output, target)

    def model_forward(self, input):
        if isinstance(input, tuple):
            return self.model(*input)
        else:
            return self.model(input)

    def residual(self, output, target):
        return output if target is None else output - target

    def kernel_forward(self, module, input, output):
        # eps is to prevent grad of sqrt() from being inf
        assert torch.is_floating_point(output), "model output have to be float type."
        eps = finfo(output.dtype).eps
        return self.kernel(output.square().sum(-1)).clamp(min=eps).sqrt()

    def loss(self, input, target, weight=None):
        output = self.model_forward(input)
        residual = self.residual(output, target, weight)
        return self.kernel(residual.sum(-1)).sum()

class consensus(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)
    def forward(self, input):
        errors=0
        for inp in input:
            error=self.nodes.Inv()@ inp
            errors=errors+ error.Log().tensor().square()
        return errors

class PoseGraph_s(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, e1,p1,e2,p2,e3,p3,nodes_other):
        #ss
        if e1 !=[]:
            node1_1=self.nodes[e1[...,0]]
            node2_1=self.nodes[e1[...,1]]
            error1 = p1.Inv() @ node1_1.Inv() @ node2_1

        #sp
        if e2 !=[]:
            node1_2=self.nodes[e2[...,0]]
            node2_2=nodes_other[e2[...,1]]
            error2 = p2.Inv() @ node1_2.Inv() @ node2_2


        #ps
        if e3 !=[]:
            node1_3=nodes_other[e3[...,0]]
            node2_3=self.nodes[e3[...,1]]
            error3 = p3.Inv() @ node1_3.Inv() @ node2_3
        
        if e1 !=[]:
            if e2!=[]:
                if e3!=[]:
                    error=torch.cat((error1,error2,error3),0)
                else:
                    error=torch.cat((error1,error2),0)
            else:
                if e3!=[]:
                    error=torch.cat((error1,error3),0)
                else:
                    error=error
        else:
            if e2!=[]:
                if e3!=[]:
                    error=torch.cat((error2,error3),0)
                else:
                    error=torch.cat((error2),0)
            else:
                if e3!=[]:
                    error=error3
                else:
                    error=None

        return error.Log().tensor()

class PoseGraph_p(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, e1,e2,e3,p1,p2,p3,nodes_other):
        if e1!=[]:
        #pp
            node1_1=self.nodes[e1[...,0]]
            node2_1=self.nodes[e1[...,1]]
            error1 = p1.Inv() @ node1_1.Inv() @ node2_1
        if e2!=[]:
        #ps
            node1_2=self.nodes[e2[...,0]]
            node2_2=nodes_other[e2[...,1]]
            error2 = p2.Inv() @ node1_2.Inv() @ node2_2
        if e3!=[]:
        #sp
            node1_3=nodes_other[e3[...,0]]
            node2_3=self.nodes[e3[...,1]]
            error3 = p3.Inv() @ node1_3.Inv() @ node2_3

        if e1 !=[]:
            if e2!=[]:
                if e3!=[]:
                    error=torch.cat((error1,error2,error3),0)
                else:
                    error=torch.cat((error1,error2),0)
            else:
                if e3!=[]:
                    error=torch.cat((error1,error3),0)
                else:
                    error=error
        else:
            if e2!=[]:
                if e3!=[]:
                    error=torch.cat((error2,error3),0)
                else:
                    error=torch.cat((error2),0)
            else:
                if e3!=[]:
                    error=error3
                else:
                    error=None

        return error.Log().tensor()

class gd_armoji(pp.optim.optimizer._Optimizer):
    def __init__(self, model, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)
        self.reject, self.reject_count = reject, 0
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model = pp.optim.optimizer.RobustModel(model, kernel, weight, auto=True)
            self.corrector = pp.optim.optimizer.Trivial()
        else:
            # manually Jacobian correction will be computed
            self.model = pp.optim.optimizer.RobustModel(model, kernel, weight, auto=False)
            self.corrector = pp.optim.optimizer.Trivial() if corrector is None else corrector

    @torch.no_grad()    
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            #gd= torch.tensor([p.grad for p in pg['params'] if p.requires_grad])
            #print(gd.size)
            #gd_sum=sum([p.grad.view(1,-1)@p.grad.view(-1,1) for p in pg['params'] if p.requires_grad])
            self.last = self.loss = self.model.loss(input, target, weight)
            self.reject_count,beta,m,mmax = 0,0.5,0,40
            up=False
            down=False
            steps=[-beta/p.grad.norm() for p in pg['params'] if p.requires_grad]
            [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
            self.loss = self.model.loss(input, target, weight)
            if self.last< self.loss:
                [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                steps=[D*0.5 for D in steps]
                self.loss=self.last
                m=m+1
                down=True
                self.last_tmp=self.last
                while down and m<=mmax:
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    if self.last_tmp<=self.loss and not up:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.loss=self.last_tmp
                        m=m+1
                    elif self.last_tmp>self.loss:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.last_tmp=self.loss
                        m=m+1
                        up=True
                    elif self.last_tmp<=self.loss and up:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.last_tmp  
                        break                      

            elif self.last>self.loss:
                [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                steps=[D*2 for D in steps]
                self.last_tmp=self.loss
                [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                self.loss = self.model.loss(input, target, weight)
                if self.last_tmp>self.loss :
                    #self.last_tmp=self.loss
                    up=True
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss=self.last_tmp

                else:
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    self.last_tmp=self.last
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1                                          
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*2 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss=self.last_tmp
                                    
        return self.loss

class gd_armoji_share(pp.optim.optimizer._Optimizer):
    def __init__(self, model, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)
        self.reject, self.reject_count = reject, 0
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model = pp.optim.optimizer.RobustModel(model, kernel, weight, auto=True)
            self.corrector = pp.optim.optimizer.Trivial()
        else:
            # manually Jacobian correction will be computed
            self.model = pp.optim.optimizer.RobustModel(model, kernel, weight, auto=False)
            self.corrector = pp.optim.optimizer.Trivial() if corrector is None else corrector
        for pg in self.param_groups:
            self.h_k=[0 for p in pg['params'] if p.requires_grad]

    @torch.no_grad()    
    def step(self, input,last_var=None,last_grad=None,neighbor_var=None,neighbor_grad=None, target=None, weight=None):
        
        for pg in self.param_groups:
            #gd= torch.tensor([p.grad for p in pg['params'] if p.requires_grad])
            #print(gd.size)
            #gd_sum=sum([p.grad.view(1,-1)@p.grad.view(-1,1) for p in pg['params'] if p.requires_grad])

            grad_f=[p.grad for p in pg['params'] if p.requires_grad]
            self.h_k = [i+j-1*k for i,j,k in zip(neighbor_grad,grad_f,self.h_k)]
            self.last = self.loss = self.model.loss(input, target, weight)
            self.reject_count,beta,m,mmax = 0,0.5,0,40
            up=False
            down=False
            steps=[-beta/p_grad.norm() for p,p_grad in zip(pg['params'],self.h_k) if p.requires_grad]
            [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
            self.loss = self.model.loss(input, target, weight)
            if self.last< self.loss:
                [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                steps=[D*0.5 for D in steps]
                self.loss=self.last
                m=m+1
                down=True
                self.last_tmp=self.last
                while down and m<=mmax:
                    [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    if self.last_tmp<=self.loss and not up:
                        [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.loss=self.last_tmp
                        m=m+1
                    elif self.last_tmp>self.loss:
                        [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.last_tmp=self.loss
                        m=m+1
                        up=True
                    elif self.last_tmp<=self.loss and up:
                        [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        self.loss=self.last_tmp  
                        break                      

            elif self.last>self.loss:
                [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                steps=[D*2 for D in steps]
                self.last_tmp=self.loss
                [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                self.loss = self.model.loss(input, target, weight)
                if self.last_tmp>self.loss :
                    #self.last_tmp=self.loss
                    up=True
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1
                    [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    self.loss=self.last_tmp

                else:
                    [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    self.last_tmp=self.last
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1                                          
                    [p.add_(-D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    steps=[D*2 for D in steps]
                    [p.add_(D*p_grad) for p,p_grad,D in zip(pg['params'],self.h_k,steps) if p.requires_grad]
                    self.loss=self.last_tmp
                                    
        return self.loss

class gd_armoji_con(pp.optim.optimizer._Optimizer):
    def __init__(self, model, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)
        self.reject, self.reject_count = reject, 0
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model = RobustModel_consensus(model, kernel, weight, auto=True)
            self.corrector = pp.optim.optimizer.Trivial()
        else:
            # manually Jacobian correction will be computed
            self.model = RobustModel_consensus(model, kernel, weight, auto=False)
            self.corrector = pp.optim.optimizer.Trivial() if corrector is None else corrector

    @torch.no_grad()    
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            #gd= torch.tensor([p.grad for p in pg['params'] if p.requires_grad])
            #print(gd.size)
            #gd_sum=sum([p.grad.view(1,-1)@p.grad.view(-1,1) for p in pg['params'] if p.requires_grad])
            self.last = self.loss = self.model.loss(input, target, weight)
            self.reject_count,beta,m,mmax = 0,0.5,0,40
            up=False
            down=False
            steps=[-beta/p.grad.norm() for p in pg['params'] if p.requires_grad]
            [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
            self.loss = self.model.loss(input, target, weight)
            if self.last< self.loss:
                [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                steps=[D*0.5 for D in steps]
                self.loss=self.last
                m=m+1
                down=True
                self.last_tmp=self.last
                while down and m<=mmax:
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    if self.last_tmp<=self.loss and not up:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.loss=self.last_tmp
                        m=m+1
                    elif self.last_tmp>self.loss:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        self.last_tmp=self.loss
                        m=m+1
                        up=True
                    elif self.last_tmp<=self.loss and up:
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.last_tmp  
                        break                      

            elif self.last>self.loss:
                [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                steps=[D*2 for D in steps]
                self.last_tmp=self.loss
                [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                self.loss = self.model.loss(input, target, weight)
                if self.last_tmp>self.loss :
                    #self.last_tmp=self.loss
                    up=True
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*2 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss=self.last_tmp

                else:
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*0.5 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss = self.model.loss(input, target, weight)
                    self.last_tmp=self.last
                    while self.last_tmp>self.loss and m<=mmax:
                        self.last_tmp=self.loss
                        [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        steps=[D*0.5 for D in steps]
                        [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                        self.loss=self.model.loss(input, target, weight)
                        m=m+1                                          
                    [p.add_(-D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    steps=[D*2 for D in steps]
                    [p.add_(D*p.grad) for p,D in zip(pg['params'],steps) if p.requires_grad]
                    self.loss=self.last_tmp
                                    
        return self.loss

class Dis_LevenbergMarquardt(pp.optim.optimizer._Optimizer):
    def __init__(self, model1, model2, solver=None, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model1.parameters(), defaults=defaults)
        #model1's variables  will be set as optimizer's parameters
        self.jackwargs = {'vectorize': vectorize, 'flatten': True}
        self.solver = Cholesky() if solver is None else solver
        self.reject, self.reject_count = reject, 0
        self.lambda_k=0
        self.weight=weight

        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model1 = pp.optim.optimizer.RobustModel(model1, kernel, auto=True)
            self.model2 = RobustModel_consensus(model2, kernel, auto=True)
            self.corrector = pp.optim.optimizer.Trivial()
        else:
            # manually Jacobian correction will be computed
            # model1 :shared variable in f
            # model2 :shared variable in phi i.e. consensus model
            self.model1 = pp.optim.optimizer.RobustModel(model1, kernel, auto=False)
            self.model2 = RobustModel_consensus(model2, kernel, auto=False)
            self.corrector = pp.optim.optimizer.Trivial() if corrector is None else corrector

    @torch.no_grad()
    def shared_variable_step(self, input1, input2, target=None, weight=None,weight2=None):
        #how to set alpha to balance the consensus quality and convergence 
        alpha=1
        thre=20
        #input2 is the list of shared variable of neighbors
        for pg in self.param_groups:
            R1 = self.model1(input1, target)
            R2 = self.model2(input2, target)
            J_f= modjac(self.model1, input=(input1, target), **self.jackwargs)
            J_phi=modjac(self.model2, input=(input2, target), **self.jackwargs)
            R_f, J_f= self.corrector(R = R1, J = J_f)
            R_phi, J_phi= self.corrector(R = R2, J = J_phi)
            loss_phi=self.model2.loss(input2, target)
            # self.last= self.loss = self.loss if hasattr(self, 'loss') \
            #                         else self.model1.loss(input1, target, weight)+self.lambda_k*loss_phi+0.5*alpha*loss_phi*loss_phi
            self.last= self.loss = self.model1.loss(input1, target)+self.lambda_k*loss_phi+0.5*alpha*loss_phi*loss_phi

            last_tmp=self.last
            down=False

            J_f_T = J_f.T.reshape((-1,) + R_f.shape)
            if weight is not None:
                J_f_T = (J_f_T.unsqueeze(-2) @ weight).squeeze(-2)
            J_f_T = J_f_T.reshape(J_f_T.shape[0], -1)

            J_phi_T = J_phi.T.reshape((-1,) + R_phi.shape)
            if weight2 is not None:
                J_phi_T = (J_phi_T.unsqueeze(-2) @ weight).squeeze(-2)
            J_phi_T = J_phi_T.reshape(J_phi_T.shape[0], -1)


            A_y, self.reject_count = J_f_T @ J_f+alpha*(J_phi_T @ R_phi.view(-1, 1)@R_phi.view(-1, 1).T @J_phi)+(loss_phi+alpha+self.lambda_k)*J_phi_T @ J_phi, 0
            A_y.diagonal().clamp_(pg['min'], pg['max'])
            #print(self.J_f.shape,self.J_phi.shape, self.R_f.shape,self.R_phi.shape)

            while last_tmp <= self.loss:
                A_y.diagonal().add_(A_y.diagonal() * pg['damping'])
                try:
                    b= -J_f_T @ R_f.view(-1, 1)-(self.lambda_k+alpha*loss_phi)*J_phi_T @ R_phi.view(-1, 1)
                    D = self.solver(A = A_y, b = b)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                #D=D*(0.5**self.reject_count)
                self.update_parameter(pg['params'], D)

                self.model2.load_state_dict(self.model1.state_dict())#replace model2's pose with model1
                loss_phi=self.model2.loss(input2, target, weight2)
                self.loss = self.model1.loss(input1, target, weight)+self.lambda_k*loss_phi+0.5*alpha*loss_phi*loss_phi

                #TRUST REGION METHOD, check the gap between the decreasement of loss and second order approximate
                #self.strategy.update(pg, last=self.last, loss=self.loss, J=J_f, D=D, R=R_f.view(-1, 1))
                self.strategy.update(pg, last=self.last, loss=self.loss, J_f=J_f, J_phi=(self.lambda_k+alpha*loss_phi)*J_phi,\
                   R_f=R_f.view(-1, 1),R_phi=R_phi.view(-1, 1), D=D)
                if last_tmp < self.loss and self.reject_count < self.reject and not down: # reject step
                    # self.strategy.update(pg, last=self.last, loss=self.loss, J_f=J_f, J_phi=(self.lambda_k+alpha*loss_phi)*J_phi,\
                    #     R_f=R_f.view(-1, 1),R_phi=R_phi.view(-1, 1), D=D)
                    self.update_parameter(params = pg['params'], step = -D)

                    self.model2.load_state_dict(self.model1.state_dict())#replace model2's pose with model1
                    loss_phi=self.model2.loss(input2, target, weight2)

                    print("loss:%7f=======>%7f ,reject_count:%2d"%(last_tmp.item(), self.loss.item(),self.reject_count))
                    self.loss, self.reject_count = last_tmp, self.reject_count + 1

                elif last_tmp > self.loss and self.reject_count < self.reject:
                    self.update_parameter(params = pg['params'], step = -D)

                    self.model2.load_state_dict(self.model1.state_dict())#replace model2's pose with model1
                    down=True
                    #loss_phi=self.model2.loss(input2, target, weight2)
                    print("loss:%7f=======>%7f ,reject_count:%2d"%(last_tmp.item(), self.loss.item(),self.reject_count))
                    last_tmp=self.loss
                    self.reject_count = self.reject_count + 1

                elif last_tmp < self.loss and down:
                    self.update_parameter(params = pg['params'], step = -D)
                    D=2*D
                    self.update_parameter(params = pg['params'], step = D)
                    self.model2.load_state_dict(self.model1.state_dict())
                    loss_phi=self.model2.loss(input2, target, weight2)
                    print("loss:%7f=======>%7f ,reject_count:%2d"%(last_tmp.item(), self.loss.item(),self.reject_count))
                    self.reject_count=0
                    self.loss=last_tmp
                    break
                else:
                    print("loss:%7f=======>%7f ,reject_count:%2d"%(self.last.item(), self.loss.item(),self.reject_count))
                    self.reject_count=0
                    break                   
            #use new common variable update lambda
            #self.model2.load_state_dict(self.model1.state_dict())#replace model2's pose with model1
            #loss_phi=self.model2.loss(input2, target, weight2)
            if self.lambda_k>=thre:
                self.lambda_k=thre
            else:
                self.lambda_k=self.lambda_k+loss_phi
            
            #use new common variable as input to update private variable          
        return self.loss



@torch.no_grad()
def plot_and_save_traj(points_list,pngname, title='', axlim=None):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    color_list=['#84A59D','#F28482','#F6BD60','#98C1D9','#3D5A80','#386D94']
    for i,points in zip(range(len(points_list)),points_list):
        points=points.detach().cpu().numpy()
        ax.plot(points[:,0], points[:,1], points[:,2],c=color_list[i],label='robot '+str(i))
    ax.legend()
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

@torch.no_grad()
def save_poses_as_tum(points1,tum_file):
    points1=points1.detach().cpu().numpy()
    with open(tum_file, 'w') as f:
        for pose in range(points1.shape[0]):
            f.write('{} {} {} {} {} {} {} {}\n'.format(pose,points1[pose][0], points1[pose][1], points1[pose][2], points1[pose][3], points1[pose][4], points1[pose][5], points1[pose][6]))

@torch.no_grad()
def PGO_G2O(robots,filepath,data):
    ids,edges,poses,infos=data.ids,data.edges.detach().cpu().numpy(),data.poses.detach().cpu().numpy(),data.infos.detach().cpu().numpy()
    with open(filepath,'w') as f:
        for rob in robots:
            num=rob.self_nodes.shape[0]
            for i,index  in zip(range(num),rob.self_nodes_id):
                line=[ "VERTEX_SE3:QUAT",str(index)]
                line.extend([str(j.item()) for j in rob.self_nodes[i]])
                line=' '.join(line)
                line+='\n'
                f.write(line)
        info_mat="1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1"
        for i in range(edges.shape[0]):
            info=infos[i]
            uptri_idx=np.triu_indices_from(info,k=0)
            info_mat=info[uptri_idx]
            info_mat=list(map(str,info_mat))
            re_mea=poses[i]
            line=["EDGE_SE3:QUAT ",str(edges[i,0].item()),str(edges[i,1].item())]
            line.extend([str(j.item()) for j in re_mea])
            line.extend(info_mat)
            line=' '.join(line)
            line=line+'\n'
            f.write(line)


class robot():
    def __init__(self,nodes_list,pri_var_list,shared_var_list,name,args):
        super().__init__()
        self.name=name
        device=args.device

        #method switch 
        self.first_order=True
        self.second_order=True

        data = G2OPGO(args.dataroot,args.dataname,args.device)
        #self.edges, self.poses, self.infos = edges,poses,infos
        self.self_nodes_id=nodes_list
        self.self_nodes=data.nodes[nodes_list].clone().to(device)
        #divide var into shared or private
        self.neighbors=[]
        self.shared_variable= []
        # for nei in range(len(shared_var_list)):
        #    # is  nei  the real  neighbor 
        #     if len(shared_var_list[nei])!=0:
        #         #add neighbor name
        #         self.shared_variable.append(nei)
        #         s_nei=[data.nodes[i].clone() for i in shared_var_list[nei]]
        #         s_nei=torch.stack(s_nei).to(device)
        #         self.shared_variable.append(s_nei)
        #     else:
        #         self.shared_variable.append(None)

        #self.shared_var_list_gathered=list(set(np.array(shared_var_list).flatten().tolist())).sort()
        shared_var_list.sort()
        self.shared_var_list_gathered= shared_var_list
        self.shared_variable = data.nodes[self.shared_var_list_gathered].clone().to(device)
        self.pri_var_list=pri_var_list
        self.private_variable =data.nodes[pri_var_list].clone().to(device)

        self.pp_edges=[]
        self.pp_poses=[]
        self.pp_infos=[]

        self.ps_edges=[]
        self.ps_poses=[]
        self.ps_infos=[]

        self.sp_edges=[]
        self.sp_poses=[]
        self.sp_infos=[]

        self.ss_edges=[]
        self.ss_poses=[]
        self.ss_infos=[]
        self.shared_infos=None
        self.private_infos=None

        self.shared_variable_index=self.shared_var_list_gathered
        self.private_variable_index=self.pri_var_list

        self.graph_shared = PoseGraph_s(self.shared_variable).to(args.device) 
        self.graph_private = PoseGraph_p(self.private_variable).to(args.device)
        self.graph_consensus = consensus(self.shared_variable).to(args.device)

        self.fs_loss_list=[]
        self.fp_loss_list=[]
        self.con_loss_list=[]


        if self.first_order:
            self.optim_share_1 = gd_armoji_share(self.graph_shared, min=1e-6, vectorize=args.vectorize)
            self.optim_consensus= gd_armoji_con(self.graph_consensus, min=1e-6, vectorize=args.vectorize)
            self.optim_private_1= gd_armoji(self.graph_private, min=1e-6, vectorize=args.vectorize)


        if self.second_order:
            solver = ppos.Cholesky()
            strategy = ppost.TrustRegion(radius=args.radius)
            strategy_1=TrustRegion_1(radius=args.radius)
            self.optim_share_2 = Dis_LevenbergMarquardt(self.graph_shared,self.graph_consensus, solver=solver, strategy=strategy_1, min=1e-6, vectorize=args.vectorize)
            self.optim_private_2= pp.optim.optimizer.LevenbergMarquardt(self.graph_private, solver=solver, strategy=strategy, min=1e-6, vectorize=args.vectorize)
        

    def receiver(self,robots):
        self.neighbors=[i for i in range(len(robots))]
        self.neighbors.remove(self.name)
        self.shared_variable_list=[]
        for rob in [robots[nei] for nei in self.neighbors]:
            self.shared_variable_list.append(rob.shared_variable.clone().detach())
        print('robot'+str(self.name)+' has neighbors consisting of '+str(self.neighbors))

    def receive_var_grad(self,neighbors):
        self.shared_variable_list=[]
        self.shared_variable_grad_list=[]
        for rob in neighbors:
            if rob.name!=self.name:
                self.shared_variable_list.append(rob.shared_variable.clone().detach())
                self.shared_variable_grad_list.append(rob.optim_share_1.h_k.copy())

    def opt_by_second_order(self):
        #get neighbors public variable
        #self.receiver(neighbors)
        #optimize public var
        self.optim_share_step()
        print(str(self.name)+' has optimized shared variables')
        #optimize private var
        self.optim_private_step()

        print(str(self.name)+' has optimized private variables')

    def opt_by_first_order(self):
        #get neighbors public variable
        #self.receiver(neighbors)
        #self.receive_var_grad(neighbors)
        #consensus step
        self.optim_consensus_step()
        #optimize public var
        self.optim_share_step1()
        print(str(self.name)+' has optimized shared variables')
        # #optimize private var
        self.optim_private_step1()
        print(str(self.name)+' has optimized private variables')
    
    def optim_consensus_step(self):
        self.optim_consensus.model.load_state_dict(self.optim_share_1.model.state_dict())
        self.optim_consensus.zero_grad()
        loss=self.optim_consensus.model.loss(input=self.shared_variable_list,target=None, weight=None)
        loss.backward()

        loss=self.optim_consensus.step(input=self.shared_variable_list,weight=None)
        self.shared_variable=self.graph_consensus.nodes.clone().detach()
        self.con_loss_list.append(loss.detach().cpu().numpy())
        return loss

    def optim_share_step1(self):
        self.optim_share_1.model.load_state_dict(self.optim_consensus.model.state_dict())
        self.optim_share_1.zero_grad()
        loss=self.optim_share_1.model.loss(input=(self.ss_edges,self.ss_poses,self.sp_edges,self.sp_poses,self.ps_edges,self.ps_poses, self.private_variable)\
            ,target=None, weight=self.shared_infos)
        loss.backward()

        neighbor_grad=[0 for i in self.shared_variable_grad_list[0]]

        num_nei=len(self.shared_variable_grad_list)
        for nei in self.shared_variable_grad_list:
            neighbor_grad=[i+(j/num_nei) for i,j in zip(neighbor_grad,nei)]
        #neighbor_grad=[0*i for i in self.shared_variable_grad_list[0]]
        
        loss=self.optim_share_1.step(input=(self.ss_edges,self.ss_poses,self.sp_edges,self.sp_poses,self.ps_edges,self.ps_poses,self.private_variable)\
            ,neighbor_grad=neighbor_grad, weight=self.shared_infos)
        self.shared_variable = self.graph_shared.nodes.clone().detach()
        self.fs_loss_list.append(loss.detach().cpu().numpy())

        index_list=[]
        var_updated=[]
        for id in self.shared_var_list_gathered:
            if self.self_nodes_id.count(id):
                index_list.append(self.self_nodes_id.index(id))
                var_updated.append(self.shared_var_list_gathered.index(id))

        self.self_nodes[index_list]=self.shared_variable[var_updated].clone().detach()
        return loss 

    def optim_private_step1(self):
        self.optim_private_1.zero_grad()
        loss=self.optim_private_1.model.loss(input=(self.pp_edges,self.ps_edges,self.sp_edges,self.pp_poses,self.ps_poses,self.sp_poses,\
            self.shared_variable),target=None, weight=self.private_infos)
        loss.backward()
        loss=self.optim_private_1.step(input=(self.pp_edges,self.ps_edges,self.sp_edges,self.pp_poses,self.ps_poses,self.sp_poses,\
            self.shared_variable), weight=self.private_infos) 
        self.private_variable =self.graph_private.nodes.clone().detach()
        self.fp_loss_list.append(loss.detach().cpu().numpy())
        index_list=[]
        var_updated=[]
        for id in self.pri_var_list:
            if self.self_nodes_id.count(id):
                index_list.append(self.self_nodes_id.index(id))
                var_updated.append(self.pri_var_list.index(id))

        self.self_nodes[index_list]=self.private_variable[var_updated].clone().detach()
        return loss               

    def optim_share_step(self):
        #step(input1=(edges, poses,nodes_index,nodes_other,nodes_other_index),input2=graph2.nodes, weight=infos1,weight2=None)
        loss=self.optim_share_2.shared_variable_step(input1=(self.ss_edges,self.ss_poses,self.sp_edges,self.sp_poses,self.ps_edges,self.ps_poses,self.private_variable),\
            input2=self.shared_variable_list,weight=self.shared_infos,weight2=None)
        # ignore infos
        #loss=self.optim_share.step(input1=(self.edges,self.poses,self.shared_variable_index,self.private_variable,self.private_variable_index),\
        #    input2=self.shared_variable_list,weight=self.shared_infos,weight2=None)
        self.shared_variable = self.graph_shared.nodes.clone().detach()
        loss_tmp=loss
        self.fs_loss_list.append(loss_tmp.detach().cpu().numpy())
        self.con_loss_list.append(self.optim_share_2.model2.loss(input=self.shared_variable_list,target=None,weight=None).detach().cpu().numpy())

        index_list=[]
        var_updated=[]
        for id in self.shared_var_list_gathered:
            if self.self_nodes_id.count(id):
                index_list.append(self.self_nodes_id.index(id))
                var_updated.append(self.shared_var_list_gathered.index(id))

        self.self_nodes[index_list]=self.shared_variable[var_updated].clone().detach()
        return loss

    def optim_private_step(self):
        loss=self.optim_private_2.step(input=(self.pp_edges,self.ps_edges,self.sp_edges,self.pp_poses,self.ps_poses,self.sp_poses,\
            self.shared_variable),weight=self.private_infos)
        # ignore infos
        # loss=self.optim_private.step(input=(self.edges,self.poses,self.shared_variable_index,self.private_variable,self.private_variable_index),\
        #     weight=self.private_infos)

        self.private_variable =self.graph_private.nodes.clone().detach()
        loss_tmp=loss
        self.fp_loss_list.append(loss_tmp.detach().cpu().numpy())

        index_list=[]
        var_updated=[]
        for id in self.pri_var_list:
            if self.self_nodes_id.count(id):
                index_list.append(self.self_nodes_id.index(id))
                var_updated.append(self.pri_var_list.index(id))

        self.self_nodes[index_list]=self.private_variable[var_updated].clone().detach()

        return loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Graph Optimization')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--radius", type=float, default=1e4, help="trust region radius")
    parser.add_argument("--save", type=str, default='/remote-home/zhaoyixian/pypose/save/torus3D2/', help="files location to save")
    parser.add_argument("--dataroot", type=str, default='/remote-home/zhaoyixian/pypose/data/', help="dataset location")
    parser.add_argument("--dataname", type=str, default='torus3D.g2o', help="dataset name")
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false', help="to save memory")
    parser.add_argument('--vectorize', action='store_true', help='to accelerate computation')
    parser.set_defaults(vectorize=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)
    #parking-garage torus3D sphere
    data = G2OPGO(args.dataroot,args.dataname,args.device)
    edges, poses, infos = data.edges, data.poses, data.infos

    num_poses=data.nodes.shape[0]
    num_rob=2
    round=30
    round_k=0
    epochs1=4
    epochs2=2
    epochs=(epochs1+epochs2)*round
    reload=True
    num_poses_per_rob=int(num_poses/num_rob)
    
    nodes_list=[]
    nodes_share_list=[]


    #save each pose id, belongs to which robot, local id，private(0),or shared(1)
    nodes_array=np.zeros((num_poses,3)) 

    #Creating mapping from global pose index to local pose index
    for rob in range(num_rob):
        start_id=rob*num_poses_per_rob
        end_id=(rob+1)*num_poses_per_rob
        if rob==num_rob-1:
            end_id=num_poses
        # for id in range(start_id,end_id):
        #     local_id=id-start_id
        #     nodes_array[id][0]=rob
        #     nodes_array[id][1]=local_id
        nodes_array[start_id:end_id,0]=rob

        nodes_id=list(range(start_id,end_id))
        nodes_list.append(nodes_id)
        # nodes_share_list.append([])
        # for neig in range(num_rob):
        #     nodes_share_list[rob].append([])

    #distinguishi which nodes is private or shared
    # for edge in edges:
    #     src=edge[0]
    #     dst=edge[1]
    #     src_robot=nodes_array[src][0]
    #     #src_id=nodes_array[src][1]
    #     dst_robot=nodes_array[dst][0]
    #     #dst_id=nodes_array[dst][1]
    #     if src_robot!=dst_robot:
    #         nodes_array[src][2]=1
    #         nodes_array[dst][2]=1
    #         # if src not in nodes_share_list[src_robot][dst_robot]:
    #         #     nodes_share_list[src_robot][dst_robot].append(src)
    #         # if dst not in nodes_share_list[src_robot][dst_robot]:
    #         #     nodes_share_list[src_robot][dst_robot].append(dst)
    #         # if src not in nodes_share_list[dst_robot][src_robot]:
    #         #     nodes_share_list[dst_robot][src_robot].append(src)
    #         # if dst not in nodes_share_list[dst_robot][src_robot]:
    #         #     nodes_share_list[dst_robot][src_robot].append(dst)
    #         if src not in nodes_share_list:
    #             nodes_share_list.append(src)
    #         if dst not in nodes_share_list:
    #             nodes_share_list.append(dst)
    src=edges[:,0].cpu().numpy()
    dst=edges[:,1].cpu().numpy()
    src_robot=nodes_array[src,0]
    dst_robot=nodes_array[dst,0]
    diff_rob_list=(src_robot!=dst_robot)
    diff_rob_list=np.where(diff_rob_list[:]==1)[0].astype(int)
    nodes_array[edges[diff_rob_list,0].cpu().numpy(),2]=1
    nodes_array[edges[diff_rob_list,1].cpu().numpy(),2]=1
    nodes_share_list=list(np.where(nodes_array[:,2]==True)[0].astype(int))


    #initial robot
    robots=[]
    for rob in range(num_rob):
        p_var=[]
        p_var_tmp=np.where(nodes_array[:,0]==rob)[0].astype(int)
        p_var=np.where(nodes_array[:,2]==0)[0].astype(int)
        p_var=list(np.intersect1d(p_var,p_var_tmp))
        # for i in range(num_poses):
        #     if nodes_array[i][0]==rob and nodes_array[i][2]==0:
        #         p_var.append(i)

        #nodes_list,private_nodes, shared_nodes_list, name, args
        robots.append(robot(nodes_list[rob],p_var,nodes_share_list,rob,args))

    if reload:
        for rob in robots:
            rob.graph_shared.load_state_dict(torch.load(args.dataroot+'rob_'+str(rob.name)+'_graph_shared.pt',map_location=args.device))
            rob.graph_private.load_state_dict(torch.load(args.dataroot+'rob_'+str(rob.name)+'_graph_private.pt',map_location=args.device))
            rob.graph_consensus.load_state_dict(torch.load(args.dataroot+'rob_'+str(rob.name)+'_graph_consensus.pt',map_location=args.device))

    #add private edges or shared_edges
    for edge,pose,info in zip(edges, poses, infos):
        src=int(edge[0])
        dst=int(edge[1])
        src_robot=int(nodes_array[src,0])
        #src_id=nodes_array[src][1]
        dst_robot=int(nodes_array[dst,0])
        #dst_id=nodes_array[dst][1]
        #private edges
        if src_robot==dst_robot:
            #pp 
            if  nodes_array[src][2]==0 and nodes_array[dst][2]==0:

                src,dst=robots[src_robot].pri_var_list.index(src),robots[src_robot].pri_var_list.index(dst)

                robots[src_robot].pp_edges.append(torch.LongTensor([src,dst]))
                robots[src_robot].pp_poses.append(pose)
                robots[src_robot].pp_infos.append(info)
                
            #ps
            elif nodes_array[src][2]==0 and nodes_array[dst][2]==1:

                src,dst=robots[src_robot].pri_var_list.index(src),robots[src_robot].shared_var_list_gathered.index(dst)

                robots[src_robot].ps_edges.append(torch.LongTensor([src,dst]))
                robots[src_robot].ps_poses.append(pose)
                robots[src_robot].ps_infos.append(info)

            #sp
            elif nodes_array[src][2]==1 and nodes_array[dst][2]==0:

                src,dst=robots[src_robot].shared_var_list_gathered.index(src),robots[src_robot].pri_var_list.index(dst)

                robots[src_robot].sp_edges.append(torch.LongTensor([src,dst]))
                robots[src_robot].sp_poses.append(pose)
                robots[src_robot].sp_infos.append(info)

            #ss
            else:
                src,dst=robots[src_robot].shared_var_list_gathered.index(src),robots[src_robot].shared_var_list_gathered.index(dst)
                #Longer_loop
                robots[src_robot].ss_edges.append(torch.LongTensor([src,dst]))
                robots[src_robot].ss_poses.append(pose)
                robots[src_robot].ss_infos.append(info)
                continue

        else:
            src_lc,dst_lc=robots[src_robot].shared_var_list_gathered.index(src),robots[src_robot].shared_var_list_gathered.index(dst)
            robots[src_robot].ss_edges.append(torch.LongTensor([src_lc,dst_lc]))
            robots[src_robot].ss_poses.append(pose)
            robots[src_robot].ss_infos.append(info)

            src_lc,dst_lc=robots[dst_robot].shared_var_list_gathered.index(src),robots[dst_robot].shared_var_list_gathered.index(dst)
            robots[dst_robot].ss_edges.append(torch.LongTensor([src_lc,dst_lc]))
            robots[dst_robot].ss_poses.append(pose)
            robots[dst_robot].ss_infos.append(info)

    for rob in robots:
        if len(rob.ss_edges)!=0:
            rob.ss_edges=torch.stack(rob.ss_edges).to(args.device)
            rob.ss_poses=torch.stack(rob.ss_poses).to(args.device)
            rob.ss_infos=torch.stack(rob.ss_infos).to(args.device)
        if len(rob.sp_edges)!=0:
            rob.sp_edges=torch.stack(rob.sp_edges).to(args.device)
            rob.sp_poses=torch.stack(rob.sp_poses).to(args.device)
            rob.sp_infos=torch.stack(rob.sp_infos).to(args.device)
        if len(rob.pp_edges)!=0:
            rob.pp_edges=torch.stack(rob.pp_edges).to(args.device)
            rob.pp_poses=torch.stack(rob.pp_poses).to(args.device)
            rob.pp_infos=torch.stack(rob.pp_infos).to(args.device)
        if len(rob.ps_edges)!=0:
            rob.ps_edges=torch.stack(rob.ps_edges).to(args.device)
            rob.ps_poses=torch.stack(rob.ps_poses).to(args.device)
            rob.ps_infos=torch.stack(rob.ps_infos).to(args.device)
        if rob.sp_infos!=[]:
            if rob.ps_infos!=[]:
                if rob.ss_infos!=[]:
                    rob.shared_infos=torch.cat((rob.ss_infos,rob.sp_infos,rob.ps_infos),0)
                    if rob.pp_infos!=[]:              
                        rob.private_infos=torch.cat((rob.pp_infos,rob.ps_infos,rob.sp_infos),0)
                    else:
                        rob.private_infos=torch.cat((rob.ps_infos,rob.sp_infos),0)
                else:
                    rob.shared_infos=torch.cat((rob.sp_infos,rob.ps_infos),0)
                    if rob.pp_infos!=[]:
                        rob.private_infos=torch.cat((rob.pp_infos,rob.ps_infos,rob.sp_infos),0)
                    else:
                        rob.private_infos=torch.cat((rob.ps_infos,rob.sp_infos),0)
            else:
                if rob.ss_infos!=[]:
                    rob.shared_infos=torch.cat((rob.ss_infos,rob.sp_infos),0)
                    if rob.pp_infos!=[]:              
                        rob.private_infos=torch.cat((rob.pp_infos,rob.sp_infos),0)
                    else:
                        rob.private_infos=rob.sp_infos
                else:
                    rob.shared_infos=rob.sp_infos
                    if rob.pp_infos!=[]:
                        rob.private_infos=torch.cat((rob.pp_infos,rob.sp_infos),0)
                    else:
                        rob.private_infos=rob.sp_infos
        else:
            if rob.ps_infos!=[]:
                if rob.ss_infos!=[]:
                    rob.shared_infos=torch.cat((rob.ss_infos,rob.ps_infos),0)
                    if rob.pp_infos!=[]:              
                        rob.private_infos=torch.cat((rob.pp_infos,rob.ps_infos),0)
                    else:
                        rob.private_infos=rob.ps_infos
                else:
                    rob.shared_infos=rob.ps_infos
                    if rob.pp_infos!=[]:
                        rob.private_infos=torch.cat((rob.pp_infos,rob.ps_infos),0)
                    else:
                        rob.private_infos=rob.ps_infos
            else:
                if rob.ss_infos!=[]:
                    rob.shared_infos=rob.ss_infos
                    if rob.pp_infos!=[]:              
                        rob.private_infos=rob.pp_infos
                    else:
                        rob.private_infos=None
                else:
                    rob.shared_infos=None
                    if rob.pp_infos!=[]:
                        rob.private_infos=rob.pp_infos
                    else:
                        rob.private_infos=None           
                    


    if robots[0].first_order:
        scheduler1 = StopOnPlateau(robots[0].optim_share_1, steps=epochs, patience=30, decreasing=1e-7, verbose=True)
    else:
        scheduler1 = StopOnPlateau(robots[0].optim_share_2, steps=epochs, patience=10, decreasing=1e-7, verbose=True)
    #schedular_step_l=[]


    name = os.path.join(args.save, args.dataname)
    # pngname = os.path.join(args.save, args.dataname+'trajectory.png')
    # axlim = plot_and_save(robot[0].graph_shared.nodes.translation(),pngname, 'pose graph of robot1')
    title = 'robot 123 at the 0 step(s)'
    points_list=[]
    for rob in robots:
        points_list.append(rob.self_nodes.translation())
    axlim = plot_and_save_traj(points_list,name+'.png', title)

    step_list=[]
    

    if robots[1].first_order:
        for rob in robots:
            rob.receive_var_grad(robots)
    else:
        for rob in robots:
            rob.receiver(robots)

    for rob in robots:
        
        
        #neighbors_names=[nei.name for nei in rob_neighbors]
        #print(rob.name+' has neighbors consisting of '+str(neighbors_names))
        if rob.first_order:
            rob.fs_loss_list.append(rob.optim_share_1.model.loss(input=(rob.ss_edges,rob.ss_poses,rob.sp_edges,rob.sp_poses,rob.ps_edges,rob.ps_poses,rob.private_variable)\
                ,target=None, weight=rob.shared_infos).detach().cpu().numpy())
            rob.fp_loss_list.append(rob.optim_private_1.model.loss(input=(rob.pp_edges,rob.ps_edges,rob.sp_edges,rob.pp_poses,rob.ps_poses,rob.sp_poses,\
                rob.shared_variable),target=None, weight=rob.private_infos).detach().cpu().numpy())
            rob.con_loss_list.append(rob.optim_consensus.model.loss(input=rob.shared_variable_list,target=None, weight=None).detach().cpu().numpy())
        else:
            l_fs=rob.optim_share_2.model1.loss(input=(rob.ss_edges,rob.ss_poses,rob.sp_edges,rob.sp_poses,rob.ps_edges,rob.ps_poses,rob.private_variable)\
                ,target=None, weight=rob.shared_infos).detach().cpu().numpy()
            l_phi=rob.optim_share_2.model2.loss(input=rob.shared_variable_list,target=None, weight=None).detach().cpu().numpy()
            lag_l=l_fs+0.1*l_phi*l_phi
            rob.fs_loss_list.append(lag_l)
            rob.con_loss_list.append(l_phi)
            rob.fp_loss_list.append(rob.optim_private_2.model.loss(input=(rob.pp_edges,rob.ps_edges,rob.sp_edges,rob.pp_poses,rob.ps_poses,rob.sp_poses,\
                rob.shared_variable),target=None, weight=rob.private_infos).detach().cpu().numpy())
    step_list.append(0)



    ### the 1st implementation: for customization and easy to extend
    while scheduler1.continual:
        for rob in robots:
            if rob.first_order and scheduler1.steps<=(round_k*(epochs2+epochs1)+epochs1-1):
                rob.receive_var_grad(robots)
            else:
                rob.receiver(robots)
        for rob in robots:
            #find neighbors
            #get neighbors public variable
            if rob.first_order and scheduler1.steps<=(round_k*(epochs2+epochs1)+epochs1-1):
                rob.opt_by_first_order()
            else:    
                rob.opt_by_second_order()



        scheduler1.step(robots[0].fs_loss_list[-1])

        if scheduler1.steps>=(round_k+1)*(epochs2+epochs1):
            round_k=round_k+1
            for rob in robots:
                rob.optim_share_1 = gd_armoji_share(rob.graph_shared, min=1e-6, vectorize=args.vectorize)
                rob.optim_consensus= gd_armoji_con(rob.graph_consensus, min=1e-6, vectorize=args.vectorize)
                rob.optim_private_1= gd_armoji(rob.graph_private, min=1e-6, vectorize=args.vectorize)

                solver = ppos.Cholesky()
                strategy = ppost.TrustRegion(radius=args.radius)
                strategy_1=TrustRegion_1(radius=args.radius)
                rob.optim_share_2 = Dis_LevenbergMarquardt(rob.graph_shared,rob.graph_consensus, solver=solver, strategy=strategy_1, min=1e-6, vectorize=args.vectorize)
                rob.optim_private_2= pp.optim.LM(rob.graph_private, solver=solver, strategy=strategy, min=1e-6, vectorize=args.vectorize)



         
        #schedular_step_l.append(scheduler1.steps)
        step_list.append(scheduler1.steps)
        if robots[0].first_order and scheduler1.steps<=(round_k*(epochs2+epochs1)+epochs1):
            if (scheduler1.steps-round_k*(epochs2+epochs1))%(epochs1/2)==0:
                name = os.path.join(args.save, args.dataname + str(scheduler1.steps))

                fs_loss_list=[rob.fs_loss_list[-1].item() for rob in robots]
                title = 'robot 123 at the %d step(s) \n with total loss %7f '%(scheduler1.steps, sum(fs_loss_list))
                points_list=[]
                for rob in robots:
                    points_list.append(rob.self_nodes.translation())
                    torch.save(rob.graph_shared.state_dict(), args.save+'rob_'+str(rob.name)+'_graph_shared.pt')
                    torch.save(rob.graph_private.state_dict(), args.save+'rob_'+str(rob.name)+'_graph_private.pt')
                    torch.save(rob.graph_consensus.state_dict(),args.save+'rob_'+str(rob.name)+'_graph_consensus.pt')
                plot_and_save_traj(points_list,name+'.png', title, axlim = axlim)
            
        else:    
            name = os.path.join(args.save, args.dataname + str(scheduler1.steps))
            # title = 'robot 123 at the %d step(s) \n with loss %7f , %7f, %7f'%(scheduler1.steps, \
            #     robots[0].fs_loss_list[-1].item(),robots[1].fs_loss_list[-1].item(),robots[2].fs_loss_list[-1].item())
            fs_loss_list=[rob.fs_loss_list[-1].item() for rob in robots]
            title = 'robot 123 at the %d step(s) \n with total loss %7f '%(scheduler1.steps, sum(fs_loss_list))
            points_list=[]
            for rob in robots:
                points_list.append(rob.self_nodes.translation())
                torch.save(rob.graph_shared.state_dict(), args.save+'rob_'+str(rob.name)+'_graph_shared.pt')
                torch.save(rob.graph_private.state_dict(), args.save+'rob_'+str(rob.name)+'_graph_private.pt')
                torch.save(rob.graph_consensus.state_dict(),args.save+'rob_'+str(rob.name)+'_graph_consensus.pt')
            plot_and_save_traj(points_list,name+'.png', title, axlim = axlim)
        # #torch.save(graph1.state_dict(), name+'.pt')

    PGO_G2O(robots,args.save+args.dataname,data)

    fig1=plt.figure(figsize=(10,10))
    color=['#84A59D','#F28482','#F6BD60','#98C1D9','#3D5A80','#386D94']
    plt.title('public variable loss for each robot')
    for i,rob in zip(range(num_rob),robots):
        plt.plot(step_list, rob.fs_loss_list,c=color[i],label=rob.name )
    plt.legend()
    plt.savefig(os.path.join(args.save, args.dataname + '_fs_loss_with_' +str(epochs))+'.png')

    fig2=plt.figure(figsize=(10,10))
    plt.title('private variable loss for each robot')
    for i,rob in zip(range(num_rob),robots):
        plt.plot(step_list, rob.fp_loss_list,c=color[i],label=rob.name )
    plt.legend()
    plt.savefig(os.path.join(args.save, args.dataname + '_fp_loss_with_' +str(epochs))+'.png') 


    fig3=plt.figure(figsize=(10,10))
    plt.title('consensus loss for each robot')
    for i,rob in zip(range(num_rob),robots):
        plt.plot(step_list, rob.con_loss_list ,c=color[i],label=rob.name )
    plt.legend()
    plt.savefig(os.path.join(args.save, args.dataname + '_consensus_loss_with_' +str(epochs))+'.png') 
    # x1=np.array(step_list)
    # y1=np.array(robot1.fs_loss_list)
    # x2=np.array(step_list)
    # y2=np.array(robot2.fs_loss_list)
    # x3=np.array(step_list)
    # y3=np.array(robot3.fs_loss_list)
    # fig1=plt.figure(figsize=(10,10))
    # plt.plot(x1,y1,'#1F131F',label=u'agent1')
    # plt.plot(x2,y2,'#6C65AA',label=u'agent2')
    # plt.plot(x3,y3,'#CA8E82',label=u'agent3')
    # plt.legend()
    # plt.savefig(os.path.join(args.save, args.dataname + '_fs_loss_with_' +str(epochs))+'.png')
    

    # x1=np.array(step_list)
    # y1=np.array(robot1.fp_loss_list)
    # x2=np.array(step_list)
    # y2=np.array(robot2.fp_loss_list)
    # x3=np.array(step_list)
    # y3=np.array(robot3.fp_loss_list)
    # fig2=plt.figure(figsize=(10,10))
    # plt.plot(x1,y1,'#1F131F',label=u'agent1')
    # plt.plot(x2,y2,'#6C65AA',label=u'agent2')
    # plt.plot(x3,y3,'#CA8E82',label=u'agent3')
    # plt.legend()
    # plt.savefig(os.path.join(args.save, args.dataname + '_fp_loss_with_' +str(epochs))+'.png')   
    # x3=np.array(steps_list3)
    # y3=np.array(loss_list3)
    # fig2=plt.figure(figsize=(7,5))
    # plt.plot(x3,y3,'g-',label=u'Consensus error between 2 agents')
    # plt.legend()
    # plt.savefig(os.path.join(args.save, args.dataname + 'consensus_loss_with_' +str(epochs))+'.png')


    for rob in robots:
        save_poses_as_tum(rob.self_nodes,args.save+'rob'+str(rob.name)+'.txt')




            