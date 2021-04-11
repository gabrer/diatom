#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from utils import weights_init_xavier, weights_init_kaiming, weights_init_normal, weights_init_sparse
from utils import gumbel_softmax

# This code comes from the AVITM Model and has been slightly adapted for DIATOM.
class VaeAvitmModel(nn.Module):
    def __init__(self, input_dimensionality, d_e, d_t, encoder_layers=2, generator_layers=0, 
                 dropout_rate=0.0, sparsity=0.0, without_decoder=False, encoder_dropout=False, generator_shortcut=False,generator_transform="softmax", device="cpu" ):

        super(VaeAvitmModel, self).__init__()
        
        self.device = device
        self.frozen = False

        self.d_i = input_dimensionality  
        self.d_e = d_e  
        self.d_t = d_t 
        self.num_of_topics = d_t 
        self.encoder_layers = encoder_layers
        self.generator_layers = generator_layers

        self.generator_transform = generator_transform   
        self.encoder_dropout     = encoder_dropout
        self.generator_shortcut  = generator_shortcut
            
        self.en1_fc  = nn.Linear(self.d_i, self.d_e)
        self.en2_fc  = nn.Linear(self.d_e, self.d_e)
        self.en_drop = nn.Dropout(dropout_rate)
        self.mean_fc = nn.Linear(self.d_e, self.d_t)
        self.mean_bn = nn.BatchNorm1d(self.d_t)
        self.logvar_fc = nn.Linear(self.d_e, self.d_t)
        self.logvar_bn = nn.BatchNorm1d(self.d_t)

        # prior mean and variance
        self.prior_mean     = torch.Tensor(1, self.d_t).fill_(0).to(device)
        self.prior_variance = 0.995
        self.prior_var      = torch.Tensor(1, self.d_t).fill_(self.prior_variance).to(device)
        self.prior_logvar   = self.prior_var.log()
        
        self.generator1 = nn.Linear(self.d_t, self.d_t)
        self.generator2 = nn.Linear(self.d_t, self.d_t)
        self.generator3 = nn.Linear(self.d_t, self.d_t)
        self.generator4 = nn.Linear(self.d_t, self.d_t)
        
        self.r_drop = nn.Dropout(dropout_rate)

        self.without_decoder = without_decoder

        # Decoder matrix
        self.de     = nn.Linear(self.d_t, self.d_i)
        self.de_bn  = nn.BatchNorm1d(self.d_i)

        # --- INIT ---
        self.init_layers_xavier()
        # Decoder initialization
        weights_init_sparse(self.de, sparsity=sparsity)
        
    
    def encoder(self, x):
        if self.encoder_layers == 1:
            pi = F.softplus(self.en1_fc(x))
            if self.encoder_dropout:
                pi = self.en_drop(pi)
        else:
            pi = F.softplus(self.en1_fc(x))
            pi = F.softplus(self.en2_fc(pi))
            # pi = gumbel_softmax(self.en1_fc(x),  temperature=0.7)
            # pi = gumbel_softmax(self.en2_fc(pi),  temperature=0.7)

            if self.encoder_dropout:
                pi = self.en_drop(pi)

        # Posterior mean and log variance
        mean   = self.mean_bn(self.mean_fc(pi))
        logvar = self.logvar_bn(self.logvar_fc(pi))
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # -- "AVITM" PyTorch version -- (https://github.com/hyqneuron/pytorch-avitm)
        var     = torch.exp(logvar)
        eps     = torch.randn_like(var)     # Noise
        h       = mean + eps * var.sqrt()   # reparameterization     
        return h, var
      
    def generator(self, h):
        if self.generator_layers == 0:
            r = h
        elif self.generator_layers == 1:
            temp = self.generator1(h)
            if self.generator_shortcut:
                r = torch.tanh(temp) + h
            else:
                r = temp
        elif self.generator_layers == 2:
            temp = torch.tanh(self.generator1(h))
            temp2 = self.generator2(temp)
            if self.generator_shortcut:
                r = torch.tanh(temp2) + h
            else:
                r = temp2
        else:
            temp = torch.tanh(self.generator1(h))
            temp2 = torch.tanh(self.generator2(temp))
            temp3 = torch.tanh(self.generator3(temp2))
            temp4 = self.generator4(temp3)
            if self.generator_shortcut:
                r = torch.tanh(temp4) + h
            else:
                r = temp4

        if self.generator_transform == 'tanh':
            return self.r_drop(torch.tanh(r))
        elif self.generator_transform == 'softmax':
            return self.r_drop(F.softmax(r, dim=1))
        elif self.generator_transform == 'relu':
            return self.r_drop(F.relu(r))
        else:  
            return self.r_drop(r)
        
    def decoder(self, r):
        p_x_given_h = F.softmax(self.de_bn(self.de(r)), dim=1)
        return p_x_given_h
    
                
    def forward(self, x):
        mean, logvar = self.encoder(x)
        h, var       = self.reparameterize(mean, logvar)
        r            = self.generator(h)
        
        if not self.without_decoder:
            p_x_given_h  = self.decoder(r)
            return (mean, logvar, var, r, p_x_given_h)    
        
        return (mean, logvar, var, r)
    
        
    def save_params(self, filename):
        torch.save(self.state_dict(), filename)
    

    def load_params(self, filename):
        self.load_state_dict(torch.load(filename))
           

    def compute_KLD(self, posterior_mean, posterior_logvar, posterior_var):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        prior_mean      = self.prior_mean.expand_as(posterior_mean)
        prior_var       = self.prior_var.expand_as(posterior_mean)
        prior_logvar    = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KLD             = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_of_topics )
        return KLD


    def loss_function_AVITM_bagOfWords(self, posterior_mean, posterior_logvar, posterior_var, p_x_given_h, DocTerm_batch, avg_loss=True):
        KLD       = self.compute_KLD(posterior_mean, posterior_logvar, posterior_var)
        nll_term  = -(DocTerm_batch * (p_x_given_h+1e-10).log()).sum(1)
        loss      = KLD + nll_term
        if avg_loss:
            loss = loss.mean()
        return (loss, nll_term, KLD)


    def loss(self, list_of_computed_params, DocTerm_batch, avg_loss=True):
        mean, logvar, var, r, p_x_given_h = list_of_computed_params
        loss, nll_term, KLD  = self.loss_function_AVITM_bagOfWords( mean, logvar, var, p_x_given_h, DocTerm_batch, avg_loss)
        return (loss, nll_term, KLD)


    def loss_function_bagOfWords(self, mean, logvar, p_x_given_h, DocTerm_batch):
        KLD      = -0.5 * torch.sum((1 + logvar - (mean ** 2) - torch.exp(logvar)),1)
        nll_term = -torch.sum( torch.mul(DocTerm_batch, torch.log(torch.mul(DocTerm_batch,p_x_given_h)+1e-32)), 1)
        # Avitm
        loss = KLD+nll_term
        # add an L1 penalty to the decoder terms
        penalty = 0       
        return loss,nll_term, KLD, penalty


    def init_layers_kaiming(self):
        weights_init_kaiming(self.en1_fc)
        weights_init_kaiming(self.en2_fc)
        weights_init_kaiming(self.mean_fc )
        weights_init_kaiming(self.logvar_fc )
        
        weights_init_kaiming(self.generator1) 
        weights_init_kaiming(self.generator2)
        weights_init_kaiming(self.generator3) 
        weights_init_kaiming(self.generator4)


    def init_layers_xavier(self):
        weights_init_xavier(self.en1_fc)
        weights_init_xavier(self.en2_fc)
        weights_init_xavier(self.mean_fc )
        weights_init_xavier(self.logvar_fc )
        
        weights_init_xavier(self.generator1) 
        weights_init_xavier(self.generator2)
        weights_init_xavier(self.generator3) 
        weights_init_xavier(self.generator4)

