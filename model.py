# Pipeline

## Import libraries
import cv2
import csv
import numpy as np
import os
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import getcwd


###FUNCTIONS
## Read csv file and append the lef, right, center file names, steering data

dataPath = 'IMG_data'
skipHeader = True
lines = []        
with open(dataPath + '/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    if skipHeader:
        next(reader, None)
    for line in reader:
        lines.append(line)

steering = []
center = []
left = []
right = []
        
for line in lines:
    if float(line[6]) < 0.1 : # Discard the image data where speed is below 0.1 - car is stationary
            continue
    steering.append(float(line[3]))
    center.append(line[0])
    left.append(line[1])
    right.append(line[2])
    
print(len(steering))

## Combine the images for left right and center and apply correction

imagepaths = []
imagepaths.extend(center)
imagepaths.extend(left)
imagepaths.extend(right)


correction = 0.25
steering_angles = []
steering_angles.extend(steering)

corrected_steering_angles_left = steering + correction*np.ones_like(steering)
steering_angles.extend(corrected_steering_angles_left)

corrected_steering_angles_right = steering - correction*np.ones_like(steering)
steering_angles.extend(corrected_steering_angles_right)

#print(imagepaths)
#print(steering_angles)
print(len(steering_angles)) # Number of the all images


'''
#This is just for testing the generator function, do not use for training
name = 'data/IMG/'+imagepaths[0].split('/')[-1]
center_image = cv2.imread(name)
center_angle = float(steering_angles[3])
print(name)
print(center_image)
print(center_angle)
'''
## Generator function from classroom. Flipping is also applied here
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'IMG_data/IMG/'+batch_sample[0].split('/')[-1] # image path
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[1]) #corresponding steering angle
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



## Do preprocessing to images
# Add lambda layer
# Apply cropping for bott0m half of the image
def PreProcessingLayers():

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model



def nVidia():

    model = PreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.30))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

### MAIN CODE

# Read the images

samples = list(zip(imagepaths, steering_angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))


# Use generator for feeding the data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
## Construct the CNN - Use Nvidia
# First preprocessing layer then Nvidia model
# Use the model
model = nVidia()
# Compile and Train the Model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples) // 32, validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples) // 32, nb_epoch=5, verbose=1)
# Save the model
model.save('model.h5')

# Print the statistics
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
# Save the model and print the statistics
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()





















