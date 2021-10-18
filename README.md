# Deep Learning Practices
A repository for different types of deep learning models that can be used as a practice. Below, you can find different models with source codes. Descriptions are directly taken from different sources. You can find the references in the final section.
## Convolutional Neural Network
Convolutional layers are the major building blocks used in convolutional neural networks.

A convolution is the simple application of a filter to an input that results in an activation. Repeated application of the same filter to an input results in a map of activations called a feature map, indicating the locations and strength of a detected feature in an input, such as an image.

The innovation of convolutional neural networks is the ability to automatically learn a large number of filters in parallel specific to a training dataset under the constraints of a specific predictive modeling problem, such as image classification. The result is highly specific features that can be detected anywhere on input images.[[1]](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

You can see the practice code for CNN with this [link](https://github.com/nuwandda/deep-learning-practices/tree/main/image_classification_with_CNN).

## Recurrent Neural Network
When it comes to sequential or time series data, traditional feedforward networks cannot be used for learning and prediction. A mechanism is required that can retain past or historic information to forecast the future values. Recurrent neural networks or RNNs for short are a variant of the conventional feedforward artificial neural networks that can deal with sequential data and can be trained to hold the knowledge about the past.[[2]](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)

You can see the practice code for RNN with this [link](https://github.com/nuwandda/deep-learning-practices/tree/main/sentiment_analysis_with_RNN).


## References
[1. How Do Convolutional Layers Work in Deep Learning Neural Networks?](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)
</br>
[2. An Introduction To Recurrent Neural Networks And The Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)
