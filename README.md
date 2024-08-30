This Python script(NumberRecognition - updated.ipynb) builds a convolutional neural network (CNN) for digit recognition using the MNIST dataset. The data, which consists of images of handwritten digits, is first preprocessed by normalizing pixel values to a [0,1] range and reshaping the images to fit the input requirements of the model. The labels are converted to one-hot encoded vectors to enable multi-class classification.

The model itself is constructed using Keras' Sequential API. It features two convolutional layers with 32 and 64 filters, respectively, each followed by max-pooling layers to reduce the spatial dimensions. The flattened output is then passed through a dropout layer to prevent overfitting, followed by a dense layer with 10 units corresponding to the digit classes, using softmax activation for classification. The model is compiled with the Adam optimizer and categorical cross-entropy loss, and is trained on the dataset with a validation split of 40%. EarlyStopping and ModelCheckpoint callbacks are implemented to halt training if improvements in validation accuracy are not seen and to save the best model.

After training, the model is evaluated on the test dataset, achieving an impressive accuracy of approximately 99.08%, with minimal loss, indicating strong performance. The model is saved to a file for future use and can be reloaded for evaluation or further training. Overall, the code demonstrates a complete workflow for developing a robust CNN model for digit recognition.
