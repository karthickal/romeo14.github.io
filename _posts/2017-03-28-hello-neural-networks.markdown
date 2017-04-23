---
layout: post
title:  "Hello Neural Networks - Handwritten Digit recognition using Keras!"
date:   2017-03-28 14:47:06 +0530
categories: [neural network, keras]
---

Neural networks are everywhere and most current products leverage them to build intelligent features. Most of these features are impossible to implement using conventional algorithmic techniques. And just like any other programming language, the neural network has a hello world program too! 

Recognizing handwritten digits is a good example problem to understand the power of neural networks. Imagine writing a program to recognize a numerical digit written on a sheet of paper. What could be the different possible ways of writing a digit? Would it be consistent across different handwritings? What about the consistency across different pens? Would it be possible to write an algorithm to detect a digit from all the possible combinations? What about extending it to alphabets then? Clearly, this is no simple problem and requires a very careful and tedious algorithm design. However, this can be easily solved by building a neural network. And modern libraries have enabled solving this problem by building a neural network in just 10-15 lines of code!
 
The hello world program of neural network recognizes handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). To keep things simple I suggest using [Keras](https://keras.io/) which runs on top of TensorFlow or Theano and is a higher level neural network API. In this post, I describe how you can build your very first neural network using python and Keras. 

>The full working source code for the neural network can be found here - [https://github.com/romeo14/hello-neural-network](https://github.com/romeo14/hello-neural-network).
The repository consists of files to visualize the MNIST data, build the neural network, train the network and recognize digits!      

# Getting Started
* * *
## Development Environment

This example uses python3, TensorFlow, Keras, numpy and other dependencies. Inspired by the starter kit on Udacity's Self-Driving Nanodegree program I have setup an image on Docker that contains all libraries and dependencies to run the network. The container also has Opencv, sklearn and pandas that you can use to run any other vision neural nets. Here are the steps to get your environment up and running -

1. Install Docker from the [official repository](https://docs.docker.com/engine/installation/)
2. Run command 'docker pull romeo14/neuralnet-toolbox' from your terminal

That's it! You now have the environment ready for development. 

## Training Data

Supervised machine learning technique involves training a model with labeled data (training data) so that it can learn a hypothesis which can then recognize the objects in real world data. The training data in our case consists of labeled images of handwritten digits and can be downloaded from the [MNIST website](http://yann.lecun.com/exdb/mnist/). The MNIST is a subset of the larger NIST data. The MNIST is a good starter database as it is already pre-processed and formatted and can be used as it is to implement learning algorithms.

# The Neural Network
* * *

## Data Visualization

It is essential that you understand how your data is formatted and represented before you implement any learning algorithms. This will usually give you insights about how your data is distributed across available labels. A disadvantage of having lots of data of a particular class compared to other classes is the bias it will introduce to the hypothesis. For example, if a neural network trains on a dataset consisting of 55000 images of dogs and 500 images of cats then it is prone to recognize dogs better than cats. In such cases, it will be prudent to generate more images of the minor class by various image augmentation techniques like flipping, shifting, zooming or rotating the existing images. 

Luckily for us, the MNIST database is clean of such nuances and can be readily used to train the model.

![Sample handwritten digits from the training set]({{ site.url }}/assets/images/hello-neural-networks/prediction_sample.png)
*Fig 1.1 - Sample handwritten digits from the training set*

The above image consists of some random digits from the dataset. Here are some different images for the same digit (3) from the dataset - 

![Sample handwritten digits of the number 3 from the training set]({{ site.url }}/assets/images/hello-neural-networks/digit_3_samples.png)
*Fig 1.2 - Sample handwritten digits of the number 3 from the training set*

The data is also equally distributed across all the labels. This will ensure our learning algorithm is not biased towards any digit and will make a more accurate prediction. 
![Histogram of the digits]({{ site.url }}/assets/images/hello-neural-networks/digits_hist.png)
*Fig 1.3 - Histogram of the digits*

The python code using matplotlib for the above visualizations can be found in the files explore.py and utils.py from the repository. Please note that the file explore.py might not display the output as intended when it is run through the docker container. It is best to run this file directly from the local environment.
 
Another important aspect to consider is the format of the labels. The labels are represented as numbers from 0 to 9 for each of the handwritten digit. In order to achieve better performance, the labels are one-hot encoded. One-hot encoding is a technique that is used to transform numerical labels to a more optimal format for classification problems. 

_example: one_hot(0) would be [1 0 0 0 0 0 0 0 0 0], one_hot(1) would be [0 1 0 0 0 0 0 0 0 0] and so on_   

## Modeling and Training

Thanks to Keras, developing and training the network is relatively simple. The neural network that we develop for this task consists of an input layer of 784 neurons, a hidden layer with 15 neurons and an output layer of 10 neurons activated by the softmax function. The file train.py contains code that creates such a neural network - 

```
train.py

# create a feedforward neural network
model = Sequential()

# add a fully connected layer to the input layer
model.add(Dense(15, input_dim=784))

# add the output layer
model.add(Dense(10))

# activate the output layer using softmax
model.add(Activation('softmax'))
```
The model is then trained with a learning rate of 0.01 and a batch size of 128 using the Stochastic Gradient Descent learning algorithm. The algorithm achieves a validation accuracy of about 92% in 25 epochs. The code to train the model is also found in the file train.py - 
```
train.py

# get the model and print summary
model = get_model()
model.summary()

# train the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.01))
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    nb_epoch=30,
    verbose=2
)
```

Here is a summary of the neural network we just built - 


<table class="simple-border">
    <tr>
        <th>Layer (type)  </th>
        <th>Output Shape</th>
        <th>Param #</th>
        <th>Connected to</th>
    </tr>
    <tr>
        <td>dense_1 (Dense)</td>
        <td>(None, 15) </td>
        <td>11775</td>
        <td>dense_input_1[0][0]</td>
    </tr>
    <tr>
        <td>dense_2 (Dense)</td>
        <td>(None, 10) </td>
        <td>160</td>
        <td>dense_1[0][0]</td>
    </tr>
    <tr>
        <td>activation_1 (Activation)</td>
        <td>(None, 10) </td>
        <td>0</td>
        <td>dense_2[0][0]</td>
    </tr>
</table>

 

>The neural network has 11935 parameters that can be tuned. Using the code in train.py you can achieve a validation accuracy of about 92% after training the model.

## Recognizing Handwritten Digits (Prediction)

Lets try predicting the digits on some of the images. For this step we randomly select some images from the testing set and run the neural network on these images.

![Selected sample for prediction]({{ site.url }}/assets/images/hello-neural-networks/prediction_sample.png)
*Fig 1.4 - Selected sample for prediction*

The selected data was [9, 5, 8, 0, 0, 8, 6, 6, 6, 8, 3, 5] and the predicted data was [9, 4, 8, 0, 0, 8, 6, 6, 6, 8, 3, 5]. The second image was predicted as 4 instead of 5, but otherwise it has done a pretty good job. The code for this can be found in the predict.py file.
 
 
 That's it! You have now built your first neural network. Congratulations!
 
 >The full working source code for the neural network can be found here - [Hello Neural Network GitHub repo](https://github.com/romeo14/hello-neural-network).

# Other Resources
* * *

For more resources and additional information check out - 

* [Michael Nielsen's book on Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits)
* [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
* [A TensorFlow tutorial on the same topic](https://www.tensorflow.org/get_started/mnist/beginners)





 

