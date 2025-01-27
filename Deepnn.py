import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

emnist_train = tfds.load('emnist/byclass', split='train', as_supervised=True)
emnist_test = tfds.load('emnist/byclass', split='test', as_supervised=True)
# Load the EMNIST dataset
# EMNIST is a dataset similar to MNIST but includes both letters and numbers.
# The dataset is split into training data and testing data.
# `as_supervised=True` loads the data as pairs of images (inputs) and labels (outputs).

# Function to preprocess the images and labels
def preprocess(image, label):
# Resize each image to 28x28 pixels, ensuring a consistent input size for the model.
    image = tf.image.resize(image, [28, 28])
    # Normalize pixel values from 0-255 to 0-1 to make training easier.
    image = tf.cast(image, tf.float32) / 255.0
    # Add an additional dimension to the image to indicate it is grayscale (1 channel).
    image = tf.expand_dims(image, axis=-1)
    return image, label  # Return the processed image and its corresponding label.

# Create a random rotation layer for data augmentation.
# Data augmentation helps the model generalize better by introducing slight variations in the training data.
random_rotation_layer = tf.keras.layers.RandomRotation(0.2)

# Function to apply additional data augmentations
def augment(image, label):
    # Randomly flip the image horizontally (left-right) to introduce variation.
    image = tf.image.random_flip_left_right(image)
    # Adjust the brightness of the image randomly.
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Adjust the contrast of the image randomly.
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Rotate the image slightly (up to ±20%) to simulate real-world variations.
    image = random_rotation_layer(image, training=True)
    return image, label  # Return the augmented image and its label.

# Prepare the training data pipeline
# The dataset is processed with the preprocessing and augmentation functions.
train_data = (emnist_train
              .map(preprocess)  # Apply preprocessing to all images.
              .map(augment)  # Apply augmentations for the training set.
              .cache()  # Cache the data in memory to speed up training.
              .shuffle(2000)  # Randomly shuffle the data to prevent patterns during training.
              .batch(64)  # Group the data into batches of 64 for efficient processing.
              .prefetch(tf.data.AUTOTUNE))  # Prefetch batches to improve training performance.

# Prepare the testing data pipeline
# Testing data is processed with only preprocessing, without augmentations.
test_data = (emnist_test
             .map(preprocess)
             .batch(64)  # Batch size for evaluation.
             .prefetch(tf.data.AUTOTUNE))  # Prefetch batches for faster evaluation.

# Define the advanced neural network model
model = tf.keras.Sequential([  # A Sequential model is a linear stack of layers.
    # First layer: Convolutional layer
    # Extracts features using 32 filters of size 3x3 and applies ReLU activation.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    # Batch normalization to stabilize training by normalizing layer outputs.
    tf.keras.layers.BatchNormalization(),
    # Downsample the image using a pooling layer to reduce computational complexity.
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    # Second convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    # Third convolutional block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    # Fourth convolutional block
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),

    # Global average pooling reduces the feature map size to a single value per filter.
    tf.keras.layers.GlobalAveragePooling2D(),
    # Dropout helps prevent overfitting by randomly setting 40% of neurons to zero.
    tf.keras.layers.Dropout(0.4),

    # Fully connected layer with 256 neurons and ReLU activation.
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # Dropout applied again to reduce overfitting.
    tf.keras.layers.Dropout(0.5),

    # Output layer for classification into 62 categories (letters and numbers).
    tf.keras.layers.Dense(62, activation='softmax')
])

# Compile the model
# Adam optimizer adjusts learning rates during training.
# Sparse Categorical Crossentropy is used for multi-class classification tasks.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler adjusts the learning rate as training progresses.
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 10))

# Train the model
# Fit the model to the training data for 15 epochs and validate on the testing data.
model.fit(train_data, epochs=15, validation_data=test_data, callbacks=[lr_scheduler])

# Save the trained model to a file for future use.
model.save('advanced_emnist_model.h5')

# Evaluate the model on the testing data to calculate loss and accuracy.
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")  # Display the test accuracy.

# Generate predictions and true labels for evaluation metrics
y_true = []
y_pred = []

for images, labels in test_data:
    # Collect true labels.
    y_true.extend(labels.numpy())
    # Predict labels for the images in the batch.
    predictions = model.predict(images)
    # Select the class with the highest predicted probability.
    predicted_classes = np.argmax(predictions, axis=1)
    # Collect predicted labels.
    y_pred.extend(predicted_classes)

# Convert the collected labels and predictions to NumPy arrays for analysis.
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print a detailed classification report showing precision, recall, and F1 score.
print("Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# Create and display a confusion matrix to visualize prediction accuracy for each class.
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')  # Use a blue color map for visualization.
plt.title("Confusion Matrix")  # Add a title to the confusion matrix plot.
plt.show()

# first epoch:     0m 73ms/step - accuracy: 0.7270 - loss: 0.9154 - val_accuracy: 0.8463 - val_loss: 0.4283 - learning_rate: 0.0010
# second epoch:    0m 95ms/step - accuracy: 0.8274 - loss: 0.4942 - val_accuracy: 0.8509 - val_loss: 0.4010 - learning_rate: 7.9433e-04
# third epoch:     0m 95ms/step - accuracy: 0.8274 - loss: 0.4942 - val_accuracy: 0.8509 - val_loss: 0.4010 - learning_rate: 7.9433e-04
# fourth epoch:    0m 86ms/step - accuracy: 0.8389 - loss: 0.4529 - val_accuracy: 0.8559 - val_loss: 0.3895 - learning_rate: 6.3096e-04
# fifth epoch:     0m 90ms/step - accuracy: 0.8461 - loss: 0.4276 - val_accuracy: 0.8567 - val_loss: 0.3841 - learning_rate: 5.0119e-04
# sixth epoch:     0m 94ms/step - accuracy: 0.8511 - loss: 0.4102 - val_accuracy: 0.8627 - val_loss: 0.3705 - learning_rate: 3.9811e-04
# ninth epoch:     0m 86ms/step - accuracy: 0.8607 - loss: 0.3782 - val_accuracy: 0.8635 - val_loss: 0.3656 - learning_rate: 1.9953e-04
# eleventh epoch:  0m 86ms/step - accuracy: 0.8641 - loss: 0.3656 - val_accuracy: 0.8653 - val_loss: 0.3642 - learning_rate: 1.2589e-04
# forteenth epoch: 0m 83ms/step - accuracy: 0.8676 - loss: 0.3528 - val_accuracy: 0.8654 - val_loss: 0.3642 - learning_rate: 6.3096e-05
