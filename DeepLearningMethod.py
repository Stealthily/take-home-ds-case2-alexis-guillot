#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:22:52 2022

@author: alexisguillot
"""
import torch.nn as nn
import torch

class SimpleNeuralNet(torch.nn.Module):
    
    
    def __init__(self, input_size, hidden_size, out_classes):
        super(SimpleNeuralNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_classes = out_classes
        
        self.layer_norm = nn.LayerNorm(input_size)
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.out_classes)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer_norm(x)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output