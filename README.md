# Emotion Detection Simulator
- This project uses Convolutional Neural Networks (CNNs) to classify emotions from facial expressions in images. The goal is to detect emotions such as happiness, sadness, anger, surprise, fear, disgust, and neutral from images. The model is built using TensorFlow and Keras, leveraging preprocessed image datasets to train and evaluate the emotion detection system.

---

## Features
- Emotion Detection: Detects a range of emotions from facial expressions.
- Convolutional Neural Networks (CNNs): Uses CNN architecture for efficient image classification.
- Image Augmentation: Implements data augmentation to improve model generalization.
- Multi-Class Classification: Classifies images into 7 emotion categories: happy, sad, angry, surprised, fear, disgust, neutral.
- Model Training: Train your own emotion detection model using available image datasets.

---

## Technologies Used
- TensorFlow: The machine learning framework for building the model.
- Keras: High-level API for defining and training neural networks.
- Python: The programming language used for this project.
- OpenCV: (Optional) For image processing tasks (if applicable).
- Matplotlib: For visualizing training progress and results.

---

## Dataset
To train the emotion detection model, you'll need a dataset of labeled images. One popular dataset for this task is the FER-2013 dataset, which contains thousands of facial expression images with emotion labels.

## Dataset Directory Structure
Organize your dataset into the following folder structure:

`
dataset/
    train/
        happy/
        sad/
        angry/
        surprised/
        fear/
        disgust/
        neutral/
    validation/
        happy/
        sad/
        angry/
        surprised/
        fear/
        disgust/
        neutral/
Replace train/ and validation/ with the actual paths to your image data.
`

---

## Model Architecture
The model is based on a standard CNN architecture with the following layers:

- Convolutional Layers: Detecting low-level and high-level features in the image.
- MaxPooling Layers: Reducing the spatial dimensions of the image.
- Fully Connected Layers: Classifying the image into one of the 7 emotion categories.
- Dropout Layer: To prevent overfitting during training.

## Training the Model
To train the emotion detection model, run the following code:
`python train.py`

1. Load the dataset.
2. Preprocess the images.
3. Train the model using the training data.
4. Validate the model on the validation set.


```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Add layers (as shown in the previous example)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocess and load data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('path_to_train_data', target_size=(48, 48), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory('path_to_validation_data', target_size=(48, 48), batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(train_generator, epochs=25, validation_data=validation_generator)
Model Evaluation
```

After training the model, you can evaluate its performance using a test dataset:

```
test_generator = validation_datagen.flow_from_directory('path_to_test_data', target_size=(48, 48), batch_size=32, class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

### Usage
Once the model is trained, you can use it to make predictions on new images:

```
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img = image.load_img('path_to_image.jpg', target_size=(48, 48))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict emotion
prediction = model.predict(img_array)
emotion = np.argmax(prediction, axis=1)

# Output predicted emotion
emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'neutral']
print(f"Predicted Emotion: {emotion_labels[emotion[0]]}")
```
