# **Kaggle Handwritten Digit Recognition**

## What is Handwritten Digit Recognition?
![](https://www.researchgate.net/profile/Hugo-Larochelle/publication/200744481/figure/fig1/AS:668968306098181@1536505881710/Samples-from-the-MNIST-digit-recognition-data-set-Here-a-black-pixel-corresponds-to-an.png)

**The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.**

<div id='content'></div>

## Index of Content

* [**Import the libraries and load the dataset**](#Chapter1)
* [**Preprocess the data**](#Chapter2)
* [**Create the model**](#Chapter3)
* [**Train the model**](#Chapter4)
* [**Evaluate on validation data**](#Chapter5)
* [**prediction on test data**](#Chapter6)
* [**submission**](#Chapter7)

## NN (Neural networks)
Neural Networks mimics the working of how our brain works. They have emerged a lot in the era of advancements in computational power.

![](https://miro.medium.com/max/1194/1*14-ce3jNHqJ5x5eb7CyTbw.png)

Deep learning is the acronym for Neural Networks, the network connected with multilayers. The layers are composited form nodes. A node is just a perception which takes an input performs some computation and then passed through a node’s activation function, to show that up to what context signal progress proceeds through the network to perform classification.

## CNN (Convolutional Neural Network)
Now let’s discuss the Convolutional Neural Networks, CNN has become famous among the recent times. CNN is part of deep, feed forward artificial neural networks that can perform a variety of task with even better time and accuracy than other classifiers, in different applications of image and video recognition, recommender system and natural language processing.

![](https://miro.medium.com/max/1144/1*22R-AyQ-oXb8Flod9PsyNw.png)

Use of CNN have spread as Facebook uses neural nets for their automatic tagging algorithms, google for photo search Amazon for their product recommendations, Pinterest for their home feed personalization and Instagram for search infrastructure. Image classification or object recognition is a problem is passing an image as a parameter and predicting whether a condition is satisfied or not (cat or not, dot or not), or the probability or most satisfying condition for an image. We are able to quickly recognize patterns, generalize from previous information and knowledge.

![](https://miro.medium.com/max/716/1*u_kP2X3t2LF_WyiLwL57Gg.png)

<div id='Chapter1'></div>

## Import the libraries and load the dataset

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

**go to [Index](#content)**

<div id='Chapter2'></div>

##  Preprocess the data

The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our neural network. The dimension of the training data is (42000,28,28). The CNN model will require one more dimension so we reshape the matrix to shape (42000,28,28,1).

**go to [Index](#content)**

<div id='Chapter3'></div>

##  Create the model
Now we will create our CNN model. A CNN model generally consists of convolutional and pooling layers. It works better for data that are represented as grid structures, this is the reason why CNN works well for image classification problems. The dropout layer is used to deactivate some of the neurons and while training, it reduces offer fitting of the model. We will then compile the model with the Adam optimizer.

## Layers of Convolutional neural network
The multiple occurring of these layers shows how deep our network is, and this formation is known as the deep neural network.

![](https://miro.medium.com/max/788/1*0NwaOkzvom6YpMZoIgWTiQ.png)


- **Input**: raw pixel values are provided as input.
- **Convolutional layer**: Input layers translates the results of neuron layer. There is need to specify the filter to be used. Each filter can only be a 5*5 window that slider over input data and get pixels with maximum intensities.
- **Rectified linear unit [ReLU] layer**: provided activation function on the data taken as an image. In the case of back propagation, ReLU function is used which prevents the values of pixels form changing.
- **Pooling layer**: Performs a down-sampling operation in volume along the dimensions (width, height).
- **Fully connected layer**: score class is focused, and a maximum score of the input digits is found.

**go to [Index](#content)**

<div id='Chapter4'></div>

## Train the model

The model.fit() function of Keras will start the training of the model. It takes the training data, validation data, epochs, and batch size.

**go to [Index](#content)**

<div id='Chapter5'></div>

## Evaluate on validation data

**go to [Index](#content)**

<div id='Chapter6'></div>

##  Prediction on Test data
We have 28,000 images in our dataset which will be used to evaluate how good our model works. The testing data was not involved in the training of the data therefore, it is new data for our model. The MNIST dataset is well balanced so we can get around 98-99% accuracy.

**go to [Index](#content)**

<div id='Chapter7'></div>

##  Submission

## **The model predict with 98.75% accuracy in kaggle Digit Recognizer competition**
