import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
csv_path = "dataset/labels/dataset.csv"  # Path to your CSV file
df = pd.read_csv(csv_path)

# Step 2: Rename columns (if necessary)
df = df.rename(columns={"image": "image_name"})  # Rename "image" to "image_name"

# Step 3: Add file extensions if missing
df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg' if not x.endswith(('.jpg', '.png')) else x)

# Step 4: Print the first few rows of the CSV file
print("CSV File Contents:")
print(df.head())

# Step 5: Print the list of files in the images folder
import os
image_files = os.listdir("dataset/images")
print("\nFiles in 'dataset/images' folder:")
print(image_files[:10])  # Print the first 10 files

# Step 6: Check if the filenames in the CSV match the files in the folder
missing_files = df[~df['image_name'].isin(image_files)]
if not missing_files.empty:
    print("\nMissing Files:")
    print(missing_files)
else:
    print("\nAll files in the CSV are present in the 'dataset/images' folder.")

# Step 7: Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 8: Define image size and batch size
img_size = (224, 224)  # Resize images to 224x224 (standard for MobileNet)
batch_size = 32  # Number of images processed at once

# Step 9: Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize validation data

# Step 10: Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="dataset/images",  # Path to your images
    x_col="image_name",  # Column with image filenames
    y_col="label",  # Column with labels
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical labels
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory="dataset/images",  # Path to your images
    x_col="image_name",  # Column with image filenames
    y_col="label",  # Column with labels
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical labels
)

# Step 11: Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 12: Train the model
epochs = 10  # Number of times the model sees the entire dataset
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs
)

# Step 13: Save the trained model
model.save("custom_fashion_model.h5")
print("Model saved as custom_fashion_model.h5")

# Step 14: Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Step 15: Plot training and validation accuracy/loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()