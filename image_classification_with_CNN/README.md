In this practice, you can learn how to train a CNN model and test it on image classification problem.
## Dataset
* To download the dataset, see this [link](https://www.kaggle.com/alessiocorrado99/animals10).
* The dataset contains about 28,000 images belonging to 10 categories: dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel and elephant.

## Model
* A set of convolutions followed by a non-linearity (ReLU in our case) and a max-pooling layer
* A linear classification layer for classifying an image into 3 categories (cats, dogs and pandas)
* The model contains around 2.23 million parameters.
* As we go down the convolutions layers, we observe that the number of channels are increasing from 3 (for RGB images) to 16, 32, 64, 128 and then to 256.
* The ReLU layer provides a non-linearity after each convolution operation.
* As the number of channels are increasing, the height and width of image is decreasing because of our max-pooling layer.
* We added Dropout in our classification layer to prevent the model from overfitting.
* We are using Adam optimizer with 0.0001 learning rate along with Cross Entropy Loss.
