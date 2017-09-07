
# coding: utf-8

from keras.models import Model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception

batch_size = 10

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rotation_range=40,
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
        'data/Images/training/',  # this is the target directory
        target_size=(299, 299),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/Images/test/',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical')


base_model = Xception(include_top=False, input_shape=(299, 299, 3))


# Freeze convolutional layers
for layer in base_model.layers[:-3]:
    layer.trainable = False



x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(114, name="categories_dense")(x)
x = Activation('sigmoid')(x)


model = Model(inputs=base_model.input, outputs=x)

model.load_weights('xception_backalayer_imgaug_notdog.h5', by_name=True)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch=32745 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=17374 // batch_size)
model.save_weights('xception_breeds_multiclass.h5')





