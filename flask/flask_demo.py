#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:04:15 2017

@author: aaron
"""

from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory, flash
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os

app_root = os.path.dirname(os.path.abspath(__file__))

upload_dir = os.path.join(app_root, 'static', 'uploads')

allowed_ext = set(['jpg', 'jpeg', 'png', 'gif'])



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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict_file',
                                    filename=filename))
    return render_template('index_orig.html')


@app.route('/predict/<filename>')
def predict_file(filename):
    return render_template('picture_show.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(upload_dir, filename)
