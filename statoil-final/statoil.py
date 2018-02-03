#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:05:40 2017

@author: leem
"""

# need to find a sensible way to import the data. 
# Would like to handle the data in something like a datframe 
# am i able to load all the data into memory

import json
import pandas as pd
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from matplotlib import pyplot as plt
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn import model_selection
from sklearn.preprocessing import normalize
from PIL import Image
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import log_loss
#import png

def createCombinedModel(optimiser):
    angle_input = Input(shape=[1], name="angle_input")
    angle_layer = Dense(1,)(angle_input)
    
    # Use a transfer model for the initial layers
    
    transfer_model = Xception(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])
    # Get the output of the last layer of transfer model. Will need to change this for each transfer model
    transfer_output = transfer_model.get_layer('block14_sepconv2_act').output
    transfer_output = GlobalMaxPooling2D()(transfer_output)
    combined_inputs = concatenate([transfer_output, angle_layer])
    
    combined_model = Dense(512, activation='relu', name="FirstFCDense")(combined_inputs)
    combined_model = Dropout(0.2)(combined_model)
    combined_model = Dense(512, activation='relu', name="SeacondFCDense")(combined_model)
    predictions = Dense(1, activation='sigmoid',name="OutputDense")(combined_model)
    
    model = Model(input=[transfer_model.input, angle_input], output =predictions)
    model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
def z_score(image_list):
    standardised_image_array = (np.array(image_list) -np.mean(image_list)) / np.std(image_list)
    return standardised_image_array.tolist()

def plotImage(standardised_list):
    standardised_image = np.array(standardised_list).reshape(75,75)
    plt.imshow(standardised_image)
    return
    
def showArrayImage(array):
    flat_array = array.flatten()
    new_array = np.asarray(map(lambda v: int(v - min(flat_array)), flat_array))
    new_array = np.asarray(map(lambda v: 255*v/new_array.max(), new_array))

    new_array.shape = (75,75)
    arrayImage = Image.fromarray(new_array, mode='L')
    arrayImage.show()
    arrayImage.save('my_not_iceberg_band_2.png')
    return

def showListImage(theList):
    image_array = np.asarray([255*(len(theList)-i[0])/len(theList) for i in sorted(enumerate(theList), key=lambda x:x[1], reverse=True)])
    pos_array = np.asarray(map(lambda v: int(v - minimum ), theList))
    minimum = min(theList)
    pos_array = np.asarray(map(lambda v: int(v - minimum ), theList))
    maximum = max(pos_array)
    image_array = np.asarray(map(lambda v: int(255*v/maximum ), pos_array))

    image_array.shape = (75,75)
    listImage = Image.fromarray(image_array, mode='L')
    listImage.show()    
    return

def frameToImagesTensor(training_frame):
    # Use z-score to standardise the image data
    standard_frame1 = training_frame['band_1'].apply(z_score)
    standard_frame2 = training_frame['band_2'].apply(z_score)

    band1 = np.asarray(map(lambda band: np.array(band).reshape(75,75),standard_frame1))
    band2 = np.asarray(map(lambda band: np.array(band).reshape(75,75),standard_frame2))
    # I take an average for the third band as the colour map was in the competiton description and looked useful
    band3 = (band1 + band2)/2
    # I want the structure of my tensor to have th echannels last as that is the default setting of keras image library
    channel_last_tensor = np.stack((band1,band2,band3),axis=3)
    return channel_last_tensor

def gen_flow_for_two_inputs(x,angle_factor,y, batch_size):
    # This function is used because the NN will only take a generator as an input
    # because we are using an genrerator to alter the main images
    # therefore the angle_factor needs to be passed in the same way a generator would
    # to do this we create two generators and take the output of the angle genrator as the second input
    gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         zoom_range = 0.2,
                         rotation_range = 360)
    gen_x = gen.flow(x,y,batch_size=batch_size,seed=0)
    gen_angle =  gen.flow(x,angle_factor,batch_size=batch_size, seed=0)
    #TODO see if i can improve on the example also need to reference this function
    while True:
        x1i = gen_x.next()
        x2i = gen_angle.next()
        yield [x1i[0], x2i[1]], x1i[1]
    

training_frame = pd.read_json("data/processed/train.json")
#testing_frame = pd.read_json("data/processed/test.json")
y_train = training_frame['is_iceberg']

avg_angle = np.mean(filter(lambda x: x != 'na' ,training_frame['inc_angle']))

# Replace the na's with the average angle
training_frame['inc_angle'] = training_frame['inc_angle'].replace('na',avg_angle)

# have to convert y_train to a numpy array as dataframe has a keras bug
#y_train = np.asarray(pd.get_dummies(training_frame['is_iceberg']))

# Converting angle to a sin as the 
angle_factor = [ np.sin(angle*np.pi/180.0) for angle in training_frame['inc_angle']]

#band_1_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_1']))
#band_2_arrays = np.asarray(map(lambda v : convertBand(v), training_frame['band_2']))
#channelled_bands = convertToTensor(training_frame,angle_factor)

# TODO experiment without horizontal, vertical and rotations
# Sun direction might affect the light see so might not want to lose that information

# Als0 need to check that having it as a channel isn't just something done for different colours as the polarization is a bit different

x_train = frameToImagesTensor(training_frame)
#x_test = frameToImagesTensor(testing_frame)
'''
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         zoom_range = 0.2,
                         rotation_range = 360)

training_flow = gen.flow(x_train)
testing_flow = gen.flow(x_test)

my_model = createModel()
my_model.fit_generator(training_flow,steps_per_epoch=24,epochs= 150)
'''

mysgd = SGD(0.01,0.5,0.0001,True)
myrmsprop = ""
modelname = ""
parametersname = ""
optimiser = "adam"
#setup constants
#best_model_filepath = "models/best%s-%s-%s-{val_acc:.2f}-{val_loss:.2f}.hdf5" % (optimiser,parametersname,modelname)
best_model_filepath = "models/final_best_model"
checkpointer = ModelCheckpoint(best_model_filepath,verbose=1, save_best_only= True)
es = EarlyStopping('val_loss', patience=10, mode="min")

#Cross validation. Stratified cross validation is done to ensure samples are representative
k = 3
folds = model_selection.StratifiedKFold(n_splits=k, shuffle=True).split(x_train,y_train)

folds_array = []
batch_size = 32
epoch_num = 1
total_train_log_loss = 0
total_test_log_loss = 0
for i in range(k):
    fold = folds.next()
    cv_training_indexes = fold[0]
    cv_testing_indexes = fold[1]
    print ("Starting fold")
    cv_x_training_samples = np.array([x_train[index] for index in cv_training_indexes])
    cv_y_training_samples = np.array([y_train[index] for index in cv_training_indexes])
   
    cv_x_testing_samples = np.array([x_train[index] for index in cv_testing_indexes])
    cv_y_testing_samples = np.array([y_train[index] for index in cv_testing_indexes])
   
    cv_train_angle_factor = np.array([angle_factor[index] for index in cv_training_indexes])
    cv_test_angle_factor = np.array([angle_factor[index] for index in cv_testing_indexes])
    
    cv_gen_test_flow = gen_flow_for_two_inputs(cv_x_testing_samples,cv_test_angle_factor,cv_y_testing_samples,batch_size)
    cv_gen_train_flow = gen_flow_for_two_inputs(cv_x_training_samples,cv_train_angle_factor,cv_y_training_samples,batch_size)
    model = createCombinedModel(optimiser)
    model.fit_generator(cv_gen_train_flow,steps_per_epoch=32, epochs=epoch_num, callbacks= [checkpointer,es], validation_data = cv_gen_test_flow,validation_steps=len(cv_test_angle_factor), verbose =1)
    
    
    model = createCombinedModel(optimiser)
    model.load_weights(best_model_filepath)
    test_pred = model.predict([cv_x_testing_samples,cv_test_angle_factor])
    test_log_loss = log_loss(cv_y_testing_samples,np.asarray(test_pred),0.0000001)
    
    train_pred = model.predict([cv_x_training_samples,cv_train_angle_factor])
    train_log_loss = log_loss(cv_y_training_samples,np.asarray(train_pred),0.0000001)
    total_train_log_loss += train_log_loss
    total_test_log_loss += test_log_loss
    print "Avg log loss: Test %03d,  Train:%03d" % (train_log_loss,test_log_loss)

avg_test_log_loss = total_test_log_loss/float(k)
avg_train_log_loss = total_train_log_loss/float(k)

print "Avg log loss: Test"
print  avg_test_log_loss
print "Train:"
print avg_train_log_loss

#immediate steps
# try different optimiser, e.g. adam optimiser
# try tensorflow gpu instead (may need to buy ram for personal laptop)
    

#immediate steps for kernal based stuff
# Need to remind myself what form the data is in when i pass it to the model
# need to convert daata into a flow and the pass into image generator
# need to pass data into the model

# Next steps:
# icebergs and ships are only a few pixels wide
# L- should therefore have the image cropped to only an area that contains the object
# L- this will reduce redundent data and speed up the algorithm see: http://elib.dlr.de/99079/2/2016_BENTES_Frost_Velotto_Tings_EUSAR_FP.pdf
# should do a 3d plot of the array too as shown in above paper
# Add in image gerneator and use it to create additional images with distortions
# use vg16 pretrained nn to help, also try the other ones available from image net
# Setup an aws instance to allow the models to be long running
# consider writing in pyspark compatible format
# Add additional conv layers
# L- research why additional conv layers would improve accuracy rather than improve performance
# could consider looking at ship wake as form of detection
# the higher the wind, the more bragg scattering, the more cluttered the ocean image will be
# the higher winds also make for improved wake detection
# clutter decreases with incidence angle

#sentinel 1 paper, can i use the following data? : At DRDC Ottawa, shore-based commercial AIS data were
#obtained in conjunction with several RADARSAT-1 and Envisat ASAR trials [32] [34] [36], for
#compilation of a database of more than 4000 validated ship signatures that may be used to
#improve models of ship RCS and its variability.

#Should consider chaning the dB into 0,255 similar to in Feature extraction of dual-pol SAR imagery for sea ice image segmentation I
# I need to find the range of the decibels,set one to 0, set the highest to 255 and apply some method to split the data across the range
# Need to check the precision of the  decibels and therefore whether this process results in loss of sig fig
    #l- from the same paper should consider perfroming the max gradient preprocessing, it's simple and was effective for them

# Should add to reading list: 


# Howell, C., Mills, J., Power, D., Youden, J., Dodge, K., Randell, C., Churchill, S., and Flett,
#D. (2006). A multivariate approach to iceberg and ship classification in HH/HV ASAR
#data. Proc. 2006 International Geoscience and Remote Sensing Symposium (IGARSS
#2006). CD-ROM proceedings. 31 July to 4 Aug. 2006, Denver, USA.


#Henschel, M.D., and Livingstone, C.E. (2006). Observation of vessel heave with airborne
#SAR. Proc. OceanSAR 2006 – The Third Workshop on Coastal and Marine Applications
#of SAR, St. John’s, NL, Canada, 23 to 25 October 2006.

#Power, D., Youden, J., Lane, K., Randell, C., and Flett, D. (2001). Iceberg detection
#capabilities of RADARSAT synthetic aperture radar. Canadian Journal of Remote Sensing,
#27(5), 476-486.

#Pond, S., and Pickard, G.L. (1983). Introductory Dynamical Oceanography, 2 nd Edition.
#Pergamon Press, Toronto.

