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

import pickle

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Model files

model = load_model('xbreeds_model.h5')
breed_to_index = np.load('breed_indices.npy').tolist()
index_to_breed = {v: k for k, v in breed_to_index.items()}
max_breed_mat = np.load('breed_max_matrix.npy')
with open('dog_vocabulary.p', 'rb') as file:
    dog_vocab = pickle.load(file)
with open('dog_lsa_matrix.npy', 'rb') as file:
    dog_lsa_mat = np.load(file)
with open('dogtime_urls.p', 'rb') as file:
    dogtime_url_dict = pickle.load(file)

generate_dog_features = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[-2].output])

dog_tfidf = TfidfVectorizer(vocabulary=dog_vocab, ngram_range = (1,2))

with open('breed_lsa.p', 'rb') as file:
    breed_lsa = pickle.load(file)

app_root = os.path.dirname(os.path.abspath(__file__))

upload_dir = os.path.join(app_root, 'static', 'uploads')

allowed_ext = set(['jpg', 'jpeg', 'png', 'gif'])

imgsize = 500, 500

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)
app.secret_key = 'seeeeecrets'
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_ext

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        desc = request.form['desc']

        if len(desc) > 10000:
            flash('Shorten your text description to less than 10000 characters.')
            return redirect(request.url)

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
            dog_txt_file = dog_file.split('.', 1)[0]+'.txt'
            with open(dog_txt_file, 'w') as file:
                file.write(desc)
            return redirect(url_for('predict_file',
                                    filename=filename))
    return render_template('landing_page.html')


@app.route('/predict/<filename>')
def predict_file(filename):

    dog_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dog_txt_file = dog_path.split('.', 1)[0]+'.txt'

    with open(dog_txt_file, 'r') as file:
        desc = file.read()

    messages = [desc]

    img = image.load_img(dog_path, target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    dog_vec = generate_dog_features([img, 0])
    best_guess = np.argmax(dog_vec)
    if best_guess == 114:
        messages.append("""Are you sure this is a dog?
                        Well, I'll give you recommendations, but I doubt they'll be
                        good.""")

    dog_text_vec = dog_tfidf.fit_transform([desc])
    dog_lsa_vec = breed_lsa.transform(dog_text_vec)

    image_similarity = cosine_similarity(max_breed_mat, dog_vec[0])
    text_similarity = cosine_similarity(dog_lsa_mat, dog_lsa_vec)
#    guess = np.argmax(image_similarity.T)
#    message += "The closest dog based on the image is "+\
#            index_to_breed[guess].split("-", 1)[1]
#    guess = np.argmax(text_similarity.T)
#    message += "<br>The closest dog based on the text is "+\
#            index_to_breed[guess].split("-", 1)[1]
    weight = 0.5
    combined_sim = weight*image_similarity + (1-weight)*text_similarity
    guesses = np.argsort(combined_sim.T)[0][::-1]

    with open(dog_txt_file, 'a') as file:
        file.write("Best guess")
        file.write(str(best_guess))

    labels = [index_to_breed[guess].split("-", 1)[0] for guess in guesses[:3]]
    breeds = [index_to_breed[guess].split("-", 1)[1] for guess in guesses[:3]]
    urls = [dogtime_url_dict[breed.lower()] for breed in breeds]
    breeds = [url.split("/")[-1].replace("-", " ").replace("_", " ").title() \
                  for url in urls]
    labels_breeds_urls = zip(labels, breeds, urls)
    return render_template('predict_image.html',
                           filename=filename, messages=messages,
                           labels_breeds_urls = labels_breeds_urls)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(upload_dir, filename)
