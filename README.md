//Implementation of a Single Layer Perceptron using the Iris dataset:
- This example uses the popular scikit-learn library to load the Iris dataset and evaluate the perceptron model.
- Iris Dataset: We load the Iris dataset from sklearn.datasets. The dataset has 3 classes (Setosa, Versicolor, Virginica), but to simplify, we use only two classes (0: Setosa, 1: Versicolor), and only the first two features (sepal length, sepal width).
- Train-Test Split: The dataset is split into training (70%) and testing (30%) sets using train_test_split.
- Standardization: The features are standardized using StandardScaler for better convergence during training.
- Model Training: The perceptron is trained using a learning rate of 0.1 for 100 epochs.
- Prediction & Evaluation: The trained perceptron predicts the labels on the test set, and the accuracy of the model is printed.

//Implementation of a Multilayer Perceptron (MLP) using TensorFlow and Keras. 
- The Iris dataset is used, which is available through sklearn, and create a neural network model with multiple layers to classify the data.
- Data Preprocessing:
      We load the Iris dataset and one-hot encode the labels (since the target is categorical with 3 classes: Setosa, Versicolor, Virginica).
      The features are standardized using StandardScaler to ensure the neural network trains effectively.
- Model Architecture:
      We use the Keras Sequential API to build a Multilayer Perceptron.
      The model has an input layer with 4 neurons (corresponding to 4 features of the Iris dataset).
      Two hidden layers are added, one with 10 neurons and another with 8 neurons, both using the ReLU activation function.
      The output layer has 3 neurons (corresponding to the three classes of Iris) with a softmax activation function to perform multi-class classification.
- Model Training:
      We compile the model using the Adam optimizer and the categorical_crossentropy loss function, which is suitable for multi-class classification problems.
      The model is trained for 100 epochs with a batch size of 10.
- Evaluation and Prediction:
      After training, the model is evaluated on the test data, and accuracy is displayed.
      Predictions are made on the test data, and the predicted classes are compared to the actual classes.
