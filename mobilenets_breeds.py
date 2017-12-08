
# coding: utf-8

import os
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.applications.mobilenet import MobileNet

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rotation_range=60,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/Images/breed_model/training/',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/Images/breed_model/testing/',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),train_generator.classes)

class_weights = dict(zip(dict(validation_generator.class_indices).values(), class_weight))

mobilenet_breeds = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3),
                             alpha=1.0, pooling='max')


# Freeze convolutional layers
for layer in mobilenet_breeds.layers[:-4]:
    layer.trainable = False



x = mobilenet_breeds.output
x = Reshape((1, 1, 1024), name='reshape_1')(x)
x = Dropout(0.25, name='dropout')(x)
x = Conv2D(114, (1, 1), padding='same', name='conv_preds')(x)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((114,), name='reshape_2')(x)

notdog_cat_model = Model(inputs=mobilenet_breeds.input, outputs=x)

notdog_cat_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy'])

filepath = "mobilenets_breeds.h5"

if os.path.exists(filepath):
    notdog_cat_model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('breed_training.log', append=True)


callbacks_list = [checkpoint, csv_logger]

notdog_cat_model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.n // batch_size,
        epochs=10,
        class_weight=class_weights,
        validation_data=validation_generator,
        validation_steps= validation_generator.n // batch_size,
        callbacks=callbacks_list)





