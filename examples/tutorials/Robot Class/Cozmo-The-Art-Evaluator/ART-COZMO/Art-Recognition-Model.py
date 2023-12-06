import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from math import ceil

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

# Set up ImageDataGenerators for training and validation
data_dir = "C:/Users/alina/Documents/GitHub/cozmo-python-sdk/examples/tutorials/Robot Class/Cozmo-The-Art-Evaluator/ART-COZMO"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # using 20% of the data for validation
)

batch_size = 32  # Define batch size

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Calculate steps_per_epoch and validation_steps
train_steps_per_epoch = ceil(train_generator.samples / batch_size)
validation_steps_per_epoch = ceil(validation_generator.samples / batch_size)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')  # Assuming 7 different artworks/categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,  # Dynamically calculated
    epochs=20,  # Adjust based on the model's performance
    validation_data=validation_generator,
    validation_steps=validation_steps_per_epoch  # Dynamically calculated
)

# Save the trained model
model.save('Art_Eval_For_Cozmo_The_Evaluator.keras')
