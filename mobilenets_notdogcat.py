
# coding: utf-8

from keras.models import Model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

batch_size = 10

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
        'data/Images/notdog_cat_model/training/',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/Images/notdog_cat_model/test/',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),train_generator.classes)

class_weights = dict(zip(dict(validation_generator.class_indices).values(), class_weight))

mobile_net_first = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3),
                             alpha=0.75, pooling='max')


# Freeze convolutional layers
for layer in mobile_net_first.layers[:-4]:
    layer.trainable = False



x = mobile_net_first.output
x = Reshape((1, 1, 768), name='reshape_1')(x)
x = Dropout(0.25, name='dropout')(x)
x = Conv2D(3, (1, 1), padding='same', name='conv_preds')(x)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((3,), name='reshape_2')(x)

notdog_cat_model = Model(inputs=mobile_net_first.input, outputs=x)

notdog_cat_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy'])

filepath = "mobilenets_notdogcat.h5"

notdog_cat_model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

notdog_cat_model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.n // batch_size,
        epochs=10,
        class_weight=class_weights,
        validation_data=validation_generator,
        validation_steps= validation_generator.n // batch_size,
        callbacks=callbacks_list)





