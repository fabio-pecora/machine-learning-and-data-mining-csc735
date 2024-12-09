import os
import numpy as np
from PIL import Image

train_path = 'Animals_DataSet/Training_Set'
validation_path = 'Animals_DataSet/Validation_Set'
test_path = 'Animals_DataSet/Test_Set'

def load_image(img_path):
    try:
        img = Image.open(img_path)  # Open the image
        img = img.convert('RGB')  # Ensure image is in RGB mode
        img = img.resize((32, 32))  # Resize to (32, 32)
        img_array = np.array(img)  # Convert to numpy array
        return img_array
        
    except Exception as e:
        print(f"Could not load image {img_path}: {e}")
        return None

def load_images_from_folder_with_labels(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder.startswith('.'):
            continue
        for filename in os.listdir(subfolder_path):
            if filename.startswith('.'):
                continue
            img_path = os.path.join(subfolder_path, filename)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                img_array = load_image(img_path)
                if img_array is not None:
                    images.append(img_array)
                    labels.append(label)

    return np.array(images), np.array(labels)

# Load images from training, validation, and test sets
train_images, train_labels = load_images_from_folder_with_labels(train_path)
validation_images, validation_labels = load_images_from_folder_with_labels(validation_path)
test_images, test_labels = load_images_from_folder_with_labels(test_path)

# Print shapes of images
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Validation images shape:", validation_images.shape)
print("Validation labels shape:", validation_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)



import numpy as np
from keras.utils import to_categorical

# Check for unique labels before adjusting
print("Unique labels in train_labels before adjustment:", np.unique(train_labels))
print("Unique labels in validation_labels before adjustment:", np.unique(validation_labels))

# Subtract 1 from all labels to shift the range to [0, 5]
train_labels = train_labels - 1
validation_labels = validation_labels - 1

# Now check if the labels are in the valid range [0, 5]
print("Unique labels in train_labels after adjustment:", np.unique(train_labels))
print("Unique labels in validation_labels after adjustment:", np.unique(validation_labels))

# Apply one-hot encoding to the adjusted labels
train_labels = to_categorical(train_labels, num_classes=6)
validation_labels = to_categorical(validation_labels, num_classes=6)

# Check the shapes of the one-hot encoded labels
print("Train labels shape after one-hot encoding:", train_labels.shape)
print("Validation labels shape after one-hot encoding:", validation_labels.shape)

# Check the unique values in the one-hot encoded labels by checking the unique rows
unique_train_labels = np.unique(train_labels, axis=0)
unique_validation_labels = np.unique(validation_labels, axis=0)

print("Unique labels in train_labels after one-hot encoding:", unique_train_labels)
print("Unique labels in validation_labels after one-hot encoding:", unique_validation_labels)




import matplotlib.pyplot as plt  # Ensure matplotlib is imported


import numpy as np  # Ensure NumPy is imported

def show_one_image_per_class(images, labels, num_classes, class_names):
    # Dictionary to store one image per class
    class_images = {}

    # Loop over images and labels
    for img, label in zip(images, labels):
        # Extract the label index from the one-hot encoded label
        label_index = np.argmax(label)  # Get the index of the '1' in the one-hot encoding

        # Add the first image for each class
        if label_index not in class_images:
            class_images[label_index] = img
        if len(class_images) == num_classes:
            break

    # Display images
    plt.figure(figsize=(12, 6))
    for i, class_label in enumerate(sorted(class_images.keys())):
        plt.subplot(1, num_classes, i + 1)  # Use `i + 1` for the subplot index
        plt.imshow(class_images[class_label])
        plt.title(class_names[class_label], fontsize=10)  # Use label directly for class_names
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Assuming 6 classes in your dataset with these names
class_names = ["butterfly", "cat", "cow", "elephant", "sheep", "squirrel"]

# Show one image for each class in the training set
NUM_CLASSES = 6
show_one_image_per_class(train_images, train_labels, NUM_CLASSES, class_names)



from sklearn.preprocessing import StandardScaler

def normalize_images(images):
    return images / 255.0  # Normalize pixel values to [0, 1]

train_images_normalized = normalize_images(train_images)
validation_images_normalized = normalize_images(validation_images)
test_images_normalized = normalize_images(test_images)


def get_class_counts(data_dir):
    class_counts = {}
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            class_counts[class_dir] = len(os.listdir(class_path))
    return class_counts


train_class_counts = get_class_counts(train_path)
validation_class_counts = get_class_counts(validation_path)

print("Train class counts:", train_class_counts)
print("Validation class counts:", validation_class_counts)
print("Class names:", class_names)



import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(class_names, class_counts, dataset_name):
    """
    Plots the class distribution as a bar graph with a legend.

    Parameters:
    - class_names: List of class names.
    - class_counts: Dictionary with class names as keys and counts as values.
    - dataset_name: Name of the dataset (e.g., "Training Set").
    """
    # Map counts based on class names
    counts = [class_counts.get(name, 0) for name in class_names]
    
    # Bar width and positions
    bar_positions = np.arange(len(class_names))
    
    # Plotting
    plt.figure(figsize=(5, 4))
    bars = plt.bar(bar_positions, counts, color='skyblue', edgecolor='black')

    # Add labels, title, and grid
    plt.xlabel("Class Names")
    plt.ylabel("Number of Images")
    plt.title(f"{dataset_name} Class Distribution")
    plt.xticks(bar_positions, class_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding legend
    legend_labels = [f"Class {i+1} ({name}: {counts[i]} images)" for i, name in enumerate(class_names)]
    plt.legend(bars, legend_labels, loc='upper right', title="Legend")
    
    # Show plot
    plt.show()


plot_class_distribution(class_names, train_class_counts, "Training Set")
plot_class_distribution(class_names, validation_class_counts, "Validation Set")


print(train_images.shape)
print(train_labels.shape)  # Should be (4200, 6)
print(validation_labels.shape)  # Should be (1200, 6)


# Check statistics of normalized validation images
print("Validation Data Stats (Normalized):")
print(f"Min: {np.min(validation_images_normalized)}")
print(f"Max: {np.max(validation_images_normalized)}")
print(f"Mean: {np.mean(validation_images_normalized)}")
print(f"Std: {np.std(validation_images_normalized)}")

# Check statistics of normalized training images
print("Train Data Stats (Normalized):")
print(f"Min: {np.min(train_images_normalized)}")
print(f"Max: {np.max(train_images_normalized)}")
print(f"Mean: {np.mean(train_images_normalized)}")
print(f"Std: {np.std(train_images_normalized)}")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,  # Reduced rotation range
    width_shift_range=0.1,  # Reduced width shift range
    height_shift_range=0.1,  # Reduced height shift range
    shear_range=0.1,  # Reduced shear range
    zoom_range=0.1,  # Reduced zoom range
    horizontal_flip=True,
    fill_mode='nearest',
)
datagen.fit(train_images_normalized)


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam 

# Define the CNN model
model_1 = models.Sequential()

# Add the first convolutional layer
model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model_1.add(layers.BatchNormalization())
model_1.add(layers.MaxPooling2D((2, 2)))

# Add the second convolutional layer
model_1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_1.add(layers.BatchNormalization())
model_1.add(layers.MaxPooling2D((2, 2)))

# Add the third convolutional layer
model_1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model_1.add(layers.BatchNormalization())
model_1.add(layers.MaxPooling2D((2, 2)))

# Add the fourth convolutional layer
model_1.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model_1.add(layers.BatchNormalization())
model_1.add(layers.GlobalAveragePooling2D())  # Use GAP instead of Flatten

# Add a fully connected (dense) layer
model_1.add(layers.Dense(128, activation='relu'))
model_1.add(layers.Dropout(0.5))  # Add dropout for regularization

# Add the output layer
model_1.add(layers.Dense(6, activation='softmax'))

# Compile the model with a tuned learning rate
model_1.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print model summary
model_1.summary()

# Train the model

history_1 = model_1.fit(
    datagen.flow(train_images_normalized, train_labels, batch_size=32),
    epochs=25,  # Increase epochs for better convergence
    validation_data=(validation_images_normalized, validation_labels),
    shuffle=True,
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history_1.history['accuracy'], label='Train Accuracy')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history_1.history['loss'], label='Train Loss')
plt.plot(history_1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define Updated Model 2
model_2 = models.Sequential()

# First Convolutional Layer with larger filters and Adam optimizer
model_2.add(layers.Conv2D(128, (5, 5), activation='linear', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))  # Larger kernel
model_2.add(layers.LeakyReLU(alpha=0.1))
model_2.add(layers.BatchNormalization())
model_2.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model_2.add(layers.Conv2D(256, (3, 3), activation='linear', kernel_regularizer=regularizers.l2(0.001)))
model_2.add(layers.LeakyReLU(alpha=0.1))
model_2.add(layers.BatchNormalization())
model_2.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer with more filters
model_2.add(layers.Conv2D(256, (3, 3), activation='linear', kernel_regularizer=regularizers.l2(0.001)))
model_2.add(layers.LeakyReLU(alpha=0.1))
model_2.add(layers.BatchNormalization())
model_2.add(layers.MaxPooling2D((2, 2)))

# Add a global average pooling layer
model_2.add(layers.GlobalAveragePooling2D())

# Fully Connected Layer with increased neurons
model_2.add(layers.Dense(512, activation='linear', kernel_regularizer=regularizers.l2(0.001)))
model_2.add(layers.LeakyReLU(alpha=0.1))  # Leaky ReLU
model_2.add(layers.Dropout(0.3))  # Reduced dropout rate

# Output layer
model_2.add(layers.Dense(6, activation='softmax'))  # Assuming 6 classes

# Compile the model with Adam optimizer
model_2.compile(optimizer=Adam(learning_rate=1e-3),  # Using Adam with a higher learning rate
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Print the model summary
model_2.summary()

# Define callbacks for better training control
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
history_2 = model_2.fit(
    datagen.flow(train_images_normalized, train_labels, batch_size=32),
    epochs=25,
    validation_data=(validation_images_normalized, validation_labels),
    shuffle=True,
    callbacks=[early_stopping, reduce_lr]
)


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history_2.history['accuracy'], label='Train Accuracy')
plt.plot(history_2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history_2.history['loss'], label='Train Loss')
plt.plot(history_2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define Model 3 - Updated Architecture
model_3 = models.Sequential()

# First Convolutional Layer
model_3.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same', input_shape=(32, 32, 3)))
model_3.add(layers.BatchNormalization())
model_3.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model_3.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
model_3.add(layers.BatchNormalization())
model_3.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer
model_3.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
model_3.add(layers.BatchNormalization())
model_3.add(layers.MaxPooling2D((2, 2)))

# Fourth Convolutional Layer
model_3.add(layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
model_3.add(layers.BatchNormalization())

# Global Average Pooling Layer
model_3.add(layers.GlobalAveragePooling2D())

# Fully Connected (Dense) Layer
model_3.add(layers.Dense(256, activation='relu'))
model_3.add(layers.Dropout(0.6))  # Increased dropout to 0.6

# Output Layer
model_3.add(layers.Dense(6, activation='softmax'))  # Assuming 6 classes

# Compile the model
model_3.compile(optimizer=Adam(learning_rate=1e-3),  # Higher initial learning rate
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Print model summary
model_3.summary()

# Define callbacks for better training control
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model
history_3 = model_3.fit(
    datagen.flow(train_images_normalized, train_labels, batch_size=32),
    epochs=25,  # Set to 25 epochs
    validation_data=(validation_images_normalized, validation_labels),
    shuffle=True,
    callbacks=[early_stopping, reduce_lr]
)


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history_3.history['accuracy'], label='Train Accuracy')
plt.plot(history_3.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history_3.history['loss'], label='Train Loss')
plt.plot(history_3.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Extract the relevant data from each history object
val_accuracy_1 = history_1.history['val_accuracy']
val_accuracy_2 = history_2.history['val_accuracy']
val_accuracy_3 = history_3.history['val_accuracy']

val_loss_1 = history_1.history['val_loss']
val_loss_2 = history_2.history['val_loss']
val_loss_3 = history_3.history['val_loss']

loss_1 = history_1.history['loss']
loss_2 = history_2.history['loss']
loss_3 = history_3.history['loss']

accuracy_1 = history_1.history['accuracy']
accuracy_2 = history_2.history['accuracy']
accuracy_3 = history_3.history['accuracy']


# Create subplots for the 4 graphs
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot validation accuracy for all three models
axs[0, 0].plot(val_accuracy_1, label='Model 1', color='blue')
axs[0, 0].plot(val_accuracy_2, label='Model 2', color='orange')
axs[0, 0].plot(val_accuracy_3, label='Model 3', color='green')
axs[0, 0].set_title('Validation Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# Plot validation loss for all three models
axs[0, 1].plot(val_loss_1, label='Model 1', color='blue')
axs[0, 1].plot(val_loss_2, label='Model 2', color='orange')
axs[0, 1].plot(val_loss_3, label='Model 3', color='green')
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Plot training loss for all three models
axs[1, 0].plot(loss_1, label='Model 1', color='blue')
axs[1, 0].plot(loss_2, label='Model 2', color='orange')
axs[1, 0].plot(loss_3, label='Model 3', color='green')
axs[1, 0].set_title('Training Loss')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Plot validation accuracy for all three models (same as first graph)
axs[1, 1].plot(accuracy_1, label='Model 1', color='blue')
axs[1, 1].plot(accuracy_2, label='Model 2', color='orange')
axs[1, 1].plot(accuracy_3, label='Model 3', color='green')
axs[1, 1].set_title('Training Accuracy')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()


# Visualizing intermediate activations

import numpy as np
import matplotlib.pyplot as plt

model_3.save('animals_modelpy.h5')

from keras.models import load_model

# Load a pre-trained model 
model = load_model('animals_modelpy.h5')
model.summary()

from tensorflow.keras.preprocessing import image

# Load and preprocess the image
img_path = '/Users/aba/Desktop/CSI_GRAD/FALL24/CSC_735_MACH/Project/Animals_DataSet/Test_Set/cow/cow (902).jpeg'
img = image.load_img(img_path, target_size=(32, 32))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

from keras.models import Model

# Create a model to fetch intermediate activations
layer_outputs = [layer.output for layer in model.layers[:10]]  # First 10 layers
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Get activations for the input image
activations = activation_model.predict(img_tensor)

# Visualize some channels from intermediate activations
def visualize_channels(activations, layer_names, channels_to_display, n_cols=5):
    for layer_activation, layer_name in zip(activations, layer_names):
        print(f"Visualizing layer: {layer_name}")
        n_channels = layer_activation.shape[-1]
        selected_channels = np.random.choice(n_channels, channels_to_display, replace=False)

        n_rows = int(np.ceil(channels_to_display / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols/1.5  , n_rows/1.5))
        
        # Remove space between subplots
        plt.subplots_adjust(wspace=0)
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < channels_to_display:
                channel_idx = selected_channels[i]
                channel_image = layer_activation[0, :, :, channel_idx]
                ax.imshow(channel_image, cmap="viridis")
            else:
                ax.axis("off")  # Turn off axes for extra subplots
            
            ax.axis("off")
        
        plt.show()
        

# Get layer names for visualization
layer_names = [layer.name for layer in model.layers[:10]] # First 10 layers


# Visualizing SOME channels
num_channels_to_display = 15 # Displaying 15 channels for each layer
visualize_channels(activations, layer_names, num_channels_to_display)