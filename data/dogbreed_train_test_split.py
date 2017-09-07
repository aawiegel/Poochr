#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:17:49 2017
Splits dog breeds according to Stanford train-test list
@author: aaron
"""

import os
from scipy.io import loadmat
import shutil

image_dir = os.path.join(os.curdir, 'Images')


test_list = loadmat('test_list.mat')
test_list_files = [item[0][0] for item in test_list['file_list']]

train_list = loadmat('train_list.mat')
train_list_files = [item[0][0] for item in train_list['file_list']]

training_dir = os.path.join(image_dir, 'training')
testing_dir = os.path.join(image_dir, 'test')

categories = list(set([item.split('/')[0] for item in test_list_files]))

to_remove = ['n02115641-dingo',
             'n02115913-dhole',
             'n02116738-African_hunting_dog',
             'n02113624-toy_poodle',
             'n02113712-miniature_poodle',
             'n02105412-kelpie',
             'n02093428-American_Staffordshire_terrier']

for remove in to_remove:
    categories.remove(remove)
    
for category in categories:
    cat_dir = os.path.join(training_dir, category)
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    
    cat_dir = os.path.join(testing_dir, category)
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)

for file in train_list_files:
    if file.split('/')[0] in to_remove:
        continue
    path1 = os.path.join(image_dir, file)
    path2 = os.path.join(training_dir, file)
    shutil.copyfile(path1, path2)

for file in test_list_files:
    if file.split('/')[0] in to_remove:
        continue
    path1 = os.path.join(image_dir, file)
    path2 = os.path.join(testing_dir, file)
    shutil.copyfile(path1, path2)
    
    
