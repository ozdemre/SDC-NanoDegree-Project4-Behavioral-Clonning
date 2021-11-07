# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./readme_pictures/ozdemre-dataset-result.png "result"
[image2]: ./readme_pictures/nvidia_CNN_model.JPG "nvidia model"
[image3]: ./readme_pictures/left_2016_12_01_13_37_38_030.jpg "nvidia model"
[image4]: ./readme_pictures/left_2021_11_01_18_40_38_127.jpg "nvidia model"
[image5]: ./readme_pictures/right_2021_11_03_19_38_11_674.jpg "nvidia model"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator, and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [Nvidia Autonomous Driving](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model as suggested from community.

The model also includes RELU and Dropout layers to introduce nonlinearity inside `nVidiaModel()` function, and the data is normalized in the model using a Keras lambda layer inside `createPreProcessingLayers()`  function. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 130-137). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 164-165). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Also, I checked my validation and training loss vs epochs graphs to make sure that I am not pushing model to overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I started batch size of 32, then increased it to 64. But since I did not see much change, I stick with 32 batch size and increased the number of epochs until overfitting occurs.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and backward driving.
Overall I had 34896 training data. (including center, left and right images)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with Nvidia CNN model with Udacity provided sample data. After 5 epochs of training I tested on the simulator. Even though CNN managed to drive the car for first two corners it failed on some tricky bends.
Changing the hyper parameters didn't help since sample data seem to be not good enough. All my efforts on tuning process ended up car being lost track and fall into a ditch or river on different corners.

Then I changed my strategy to collect much more data for training, apply augmentation techniques and train the model with high number of epochs as long as no overfitting observed.
In addition to the Udacity provided data, I collected 3 laps of smooth driving, 1 lap of backward driving, and 1 lap of recovery driving.
I also applied the correction factor for left and right side of cameras. There is no right answer for correction factor, but I ended up being satisfied wih 0.25.

Since base nVidia model is prone to overfitting I added additional Dropout layers to prevent overfitting. I also introduced additional ReLu activations in order to increase nonlinearity.

Cropping was necessary since we do not need upper parts of the image where road curves can not be seen.

After my modifications, the vehicle is able to drive autonomously around the track without leaving the road. I only tested on first track since I didn't collect any data for second track.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with additional Dropout layers and ReLu activations. 

Here is a visualization of the architecture.


```
Layer (type)                                          
================================
Lambda Normalize                   
________________________________
Cropping2D (50,20)                                          
________________________________
Convolution2D(24,5,5) - ReLu                
________________________________
Convolution2D(36,5,5) - ReLu            
________________________________
Convolution2D(48,5,5) - ReLu         
________________________________
Convolution2D(64,3,3) - ReLu            
________________________________
Dropout (0.3)                   
________________________________
Convolution2D(64,3,3) - ReLu            
________________________________
Flatten                         
________________________________
Dense(100) - ReLu                   
________________________________
Dropout (0.3)                   
________________________________
Dense(50) - ReLu                                    
________________________________
Dense(10) - ReLu                                    
________________________________
Dense(1) - ReLu                                   
================================
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I collected 3 laps of smooth driving, 1 lap of backward driving, and 1 lap of recovery driving. After realizing, I am having problems with the corner -just after the bridge- I collected additional data for that particular corner to train the model better. 


After the collection process, I had 34896 number of data points. I then preprocessed this data by normalizing it with Lambda layer and cropping it from lower part.
Here are some examples of training data:

Backward Driving

![alt text][image3]

Forward Driving

![alt text][image4]

Left Camera Example

![alt text][image5]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 as I started to get satisfying results. More can be used by considering overfitting. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
Here is the training output where training and validation loss vs epochs are given.

![alt text][image1]


After training here is [video](https://youtu.be/0_o6oSI1pTM) where my model driving the vehicle on first track autonomously.

