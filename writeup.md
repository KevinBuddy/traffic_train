#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_dis]: GTSRB/distribution.png 
[image_ori]: GTSRB/origin.png 
[image_grayscale]: GTSRB/grayscale.png 
[image1]: GTSRB/00000.png 
[image2]: GTSRB/00001.png 
[image3]: GTSRB/00002.png 
[image4]: GTSRB/00003.png 
[image5]: GTSRB/00004.png 

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![image_dis]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
As a first step,I blured 5000 images of the training set in order to prevent overfiting.

As a second step, I decided to convert the images to grayscale because the grayscale image has 1 channel,so I don't need to change the conv1_w and x structure.

Here is an example of a traffic sign image before and after grayscaling.

![][image_ori]![][image_grayscale]

As a last step, I normalized the image data in order to keep the gradient of the neural network from too big or too small. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten			    | outputs 400   								|
| Fully connected		| outputs 120       							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Fully connected		| outputs 43    								|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer.The batch size is 30,the number of epochs is 10,the learning rate is 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.936
* training set accuracy of 0.993
* test set accuracy of 0.912
  first,I've preprocessed the input image,make it grayscale and normalization.Then I found that the Validation Accuracy rised up to 0.899 and began to decline to 0.85,then again rised up to 0.901,then again declined.Meanwhile,the accuracy of training set is 0.995,that is overfit.So I've blured some image of the training set and the Validation Accuracy has rised up to 0.936.  

---

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)	| Speed limit (100km/h)							|
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| No passing			| No passing									|
| Ahead only			| Ahead only					 				|
| Children crossing		| Children crossing    							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.912

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit (100km/h) sign (probability of 0.99), and the image does contain a Speed limit (100km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99879122e-01        |Speed limit (100km/h) 							|
| 9.33707561e-05        |Roundabout mandatory                           |
| 2.35775169e-05        |Speed limit (80km/h)                           |
| 3.87899672e-06        |Speed limit (120km/h)                          |
| 2.54624286e-08        |Speed limit (50km/h)                           |

For the second image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 0.99), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| 9.99999881e-01        |Speed limit (60km/h)                           |
| 7.87869965e-08        |Speed limit (80km/h)                           |
| 2.83151147e-09        |Speed limit (50km/h)                           |
| 1.52822958e-12        |Turn right ahead                               |
| 1.37988145e-13        |End of all speed and passing limits            |

For the third image, the model is relatively sure that this is a No passing sign (probability of 1.00), and the image does contain a No passing sign. The top five soft max probabilities were

| 1.00000000e+00        |No passing                                     |
| 2.97774315e-26        |End of no passing                              |
| 1.03198067e-28        |No passing for vehicles over 3.5 metric tons   |
| 1.36813337e-35        |Ahead only                                     |
| 4.29331303e-36        |Priority road                                  |

For the forth image, the model is relatively sure that this is a Ahead only sign (probability of 0.98), and the image does contain a Ahead only sign. The top five soft max probabilities were

| 9.86312926e-01        |Ahead only                                     |
| 1.36623308e-02        |Turn left ahead                                |
| 2.38123394e-05        |Turn right ahead                               |
| 9.04713431e-07        |No passing for vehicles over 3.5 metric tons   |
| 2.48407730e-08        |Yield                                          |

For the fifth image, the model is relatively sure that this is a Children crossing sign (probability of 1.00), and the image does contain a Children crossing sign. The top five soft max probabilities were

| 1.00000000e+00        |Children crossing                              |
| 8.42977688e-09        |Beware of ice/snow                             |
| 1.43892190e-10        |Bicycles crossing                              |
| 2.79742549e-12        |Dangerous curve to the right                   |
| 2.06653447e-13        |Road narrows on the right                      |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I thought it's border.