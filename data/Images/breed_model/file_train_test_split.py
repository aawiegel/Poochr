#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:56:32 2017

@author: aaron
"""

import os
import random
import shutil

def file_train_test_split(class_name, random_seed=42, split=0.7):
    """
    For a given class name, splits files into training and test data
    
    Inputs
    ========
    class_name: the name of the class directory to split
    
    random_seed: random seed to use in train test split (default 42)
    
    split: fraction of data to split into train and test data
    
    Outputs
    ========
    None, but shuffles files in function
    """
    
    class_dir = os.path.join(os.curdir, class_name)
    
    class_files = os.listdir(class_dir)
    
    random.seed(random_seed)
    
    random.shuffle(class_files)
    
    class_split_index = int(len(class_files) * split)
    class_training = class_files[:class_split_index]
    class_testing = class_files[class_split_index:]
    
    training_dir = os.path.join(os.curdir, 'training')
    testing_dir = os.path.join(os.curdir, 'testing')

    
    
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    
    os.makedirs(os.path.join(training_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(testing_dir, class_name), exist_ok=True)
    
    for file in class_training:
        path1 = os.path.join(class_dir, file)
        path2 = os.path.join(training_dir, class_name, file)
        shutil.copyfile(path1, path2)
        
    for file in class_testing:
        path1 = os.path.join(class_dir, file)
        path2 = os.path.join(testing_dir, class_name, file)
        shutil.copyfile(path1, path2)

directories = os.listdir()

directories.remove('file_train_test_split.py')

if os.path.exists(os.path.join(os.path.curdir, 'training')):
    directories.remove('training')
if os.path.exists(os.path.join(os.path.curdir, 'testing')):
    directories.remove('testing')


for class_type in directories:
    print(f"Splitting {class_type}")
    file_train_test_split(class_type)
    
        