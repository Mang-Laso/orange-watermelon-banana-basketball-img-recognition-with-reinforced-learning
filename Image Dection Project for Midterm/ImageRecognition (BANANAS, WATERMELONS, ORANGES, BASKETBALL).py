#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras import datasets, layers, models

class ImageClassifierApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Classifier App")

        self.class_names = {'banana': 0, 'watermelon': 1, 'orange': 2, 'basketball': 3}
        self.epochs = 20
        self.images = None
        self.labels = None
        self.model = self.build_model()
        self.load_dataset()

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.predict_button = tk.Button(self.root, text="Predict Image", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.correct_buttons = []
        for class_name in self.class_names:
            button = tk.Button(self.root, text=class_name.capitalize(), command=lambda class_name=class_name: self.correct_prediction(class_name))
            self.correct_buttons.append(button)
            button.pack(side=tk.LEFT, padx=10)

        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.prediction_label = tk.Label(self.root, text="")
        self.prediction_label.pack(pady=10)

        self.root.mainloop()


    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))  # 4 output classes
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def load_dataset(self):
        dataset_dir = "./"
        images = []
        labels = []
        for class_name, label in self.class_names.items():
            class_dir = os.path.join(dataset_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # Read image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                # Resize image to (32, 32)
                image = cv2.resize(image, (32, 32))
                # Normalize pixel values
                image = image / 255.0
                # Append image and label to lists
                images.append(image)
                labels.append(label)

        self.images = np.array(images).reshape(-1, 32, 32, 3)  # Reshape images for color
        self.labels = np.array(labels, dtype='int32')  # Convert labels to int32

        # Shuffle images and labels
        permutation = np.random.permutation(len(images))
        self.images = self.images[permutation]
        self.labels = self.labels[permutation]
        
        # Train the model with the loaded dataset
        self.train_model()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg *.png"),))
        if self.image_path:
            self.load_and_display_image()
            self.predict_image()  # Remove the argument passed to predict_image

    def load_and_display_image(self):
        print("Loading image:", self.image_path)
        color_img = cv2.imread(self.image_path)  # Read image in color mode
        if color_img is None:
            print("Error: Failed to load image.")
            return
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert color image to RGB format
        color_img = cv2.resize(color_img, (300, 300))  # Resize to match model input size
        color_img = color_img / 255.0  # Normalize pixel values
        self.pil_image = Image.fromarray((color_img * 255).astype(np.uint8))  # Display color image
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        print("Color image displayed successfully.")

    def predict_image(self):
        if self.image_path:
            img = cv2.imread(self.image_path)  # Read image in color mode
            img = cv2.resize(img, (32, 32))  # Resize to match model input size
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            prediction = self.model.predict(img)
            predicted_class_index = np.argmax(prediction)
            predicted_class = list(self.class_names.keys())[predicted_class_index]
            probability = prediction[0][predicted_class_index]
            self.prediction_label.config(text=f"Predicted Class: {predicted_class.capitalize()}, Probability: {probability:.2f}")
        else:
            messagebox.showerror("Error", "Please upload an image first.")

    def correct_prediction(self, correct_class):
        if self.image_path:
            # Get the destination folder for the correct class
            destination_folder = os.path.join(".", correct_class)

            # Copy the uploaded image to the destination folder
            shutil.copy(self.image_path, os.path.join(destination_folder, os.path.basename(self.image_path)))
            messagebox.showinfo("Image Copied", f"Image copied to {correct_class} folder.")

            # Retrain the model using the updated dataset
            self.load_dataset()  # Reload the dataset with the newly added image
        else:
            messagebox.showwarning("No Image", "No image has been uploaded yet.")


    def train_model(self):
        self.model.fit(self.images, self.labels, epochs=self.epochs, validation_split=0.2)
        messagebox.showinfo("Retraining", "Model retrained.")

if __name__ == "__main__":
    app = ImageClassifierApp()


# In[ ]:




