#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:17:49 2017

@author: aaron
"""

import os
import random
import shutil

dog_dir = os.path.join(os.curdir, 'dog')
not_dog_dir = os.path.join(os.curdir, 'not_dog')

dog_files = os.listdir(dog_dir)
not_dog_files = os.listdir(not_dog_dir)

random.seed(42)
random.shuffle(dog_files)
random.seed(42)
random.shuffle(not_dog_files)

split = 0.6
dog_split_index = int(len(dog_files) * split)
dog_training = dog_files[:dog_split_index]
dog_testing = dog_files[dog_split_index:]

not_dog_split_index = int(len(not_dog_files) * split)
not_dog_training = not_dog_files[:not_dog_split_index]
not_dog_testing = not_dog_files[not_dog_split_index:]

training_dir = os.path.join(os.curdir, 'training')
testing_dir = os.path.join(os.curdir, 'test')


if not os.path.exists(training_dir):
    os.makedirs(training_dir)
    #os.makedirs(os.path.join(training_dir, "dog"))
    os.makedirs(os.path.join(training_dir, "not_dog"))

if not os.path.exists(testing_dir):
    os.makedirs(testing_dir)
    #os.makedirs(os.path.join(testing_dir, "dog"))
    os.makedirs(os.path.join(testing_dir, "not_dog"))

#for file in dog_training:
#    path1 = os.path.join(dog_dir, file)
#    path2 = os.path.join(training_dir, "dog", file)
#    shutil.copyfile(path1, path2)
    
#for file in dog_testing:
#    path1 = os.path.join(dog_dir, file)
#    path2 = os.path.join(testing_dir, "dog", file)
#    shutil.copyfile(path1, path2)
    
for file in not_dog_training:
    path1 = os.path.join(not_dog_dir, file)
    path2 = os.path.join(training_dir, "not_dog", file)
    shutil.copyfile(path1, path2)
    
for file in not_dog_testing:
    path1 = os.path.join(not_dog_dir, file)
    path2 = os.path.join(testing_dir, "not_dog", file)
    shutil.copyfile(path1, path2)