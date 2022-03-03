#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:39:47 2022

@author: alexisguillot
"""
import yaml
import sys
from extra_funcs import train, load_data, evaluate

config_file = sys.argv[1]

with open(config_file, "r") as f:
    config = yaml.load(f, yaml.SafeLoader)

kwargsModel = config["kwargsModel"]


if __name__ == "__main__":
    print(kwargsModel)
    data = load_data(num_samples = -1)
    
    predictions = train(data, **kwargsModel)
    
    evaluate(predictions[0],predictions[1])
    