#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:04:15 2017

@author: aaron
"""
import os

import numpy as np

from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory, flash
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

import PIL

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

# Model files

model = load_model('../xbreeds_model.h5')
breed_to_index = np.load('../breed_indices.npy').tolist()
index_to_breed = inv_map = {v: k for k, v in breed_to_index.items()}
max_breed_mat = np.load('../breed_max_matrix.npy')

generate_dog_features = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[-2].output])


app_root = os.path.dirname(os.path.abspath(__file__))

upload_dir = os.path.join(app_root, 'static', 'uploads')

allowed_ext = set(['jpg', 'jpeg', 'png', 'gif'])

imgsize = 500, 500

app = Flask(__name__)
Bootstrap(app)
app.secret_key = 'seeeeecrets'
app.config['UPLOAD_FOLDER'] = upload_dir

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_ext

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('A file has not been selected.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dog_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dog_file)
            return redirect(url_for('predict_file',
                                    filename=filename))
    return render_template('index_orig.html')


@app.route('/predict/<filename>')
def predict_file(filename):
    dog_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = image.load_img(dog_path, target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    dog_vec = generate_dog_features([img, 0])
    best_guess = np.argmax(dog_vec)
    if best_guess == 114:
        message = "Are you sure this is a dog?"
    else:
        message = "This looks like it may be a "+\
            index_to_breed[best_guess].split("-", 1)[1]
    return render_template('picture_show.html', filename=filename, message=message)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(upload_dir, filename)
