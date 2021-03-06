# -*- coding: utf-8 -*-
"""Pretrained_Resnet

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_astaOCMI1HyRGmMVbVui6_leQsp7y7w
"""

pre_model = load_model('model/model.hdf5', compile=False) # Loading the pretrained model
pre_model.compile(loss='categorical_crossentropy', optimizer=Adam())

def hybrid_model(pre_model):
    flat = pre_model.layers[-2].output
    new_model = Model(inputs = pre_model.layers[0].output, outputs=flat)
    
    model = tf.keras.models.Sequential()
    model.add(Conv1D(64, 7, padding = 'valid', strides = 2,  activation = 'relu', input_shape = (32823,6)))
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 5, padding = 'valid', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(16, 5, padding = 'valid', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(12, 5, padding = 'valid', strides = 1, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(new_model)
    model.add(Dense(7, 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics = ['accuracy'])
    return model

"""## Pretrained Model """

!wget https://zenodo.org/record/3765717/files/model.zip?download=1
!mv model.zip?download=1 model.zip
!unzip -qq model.zip