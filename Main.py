import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.cluster import KMeans
import webcolors
from webcolors import hex_to_name
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define a function to extract the dominant color from an image
def extract_dominant_color(image):
    # Convert the image to a 2D array of pixels
    pixels = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    # Use K-Means clustering to find the dominant color
    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

# Define a function to handle the file selector dialog
def load_image():
    # Open a file selector dialog and get the selected file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    # Load the image and extract the dominant color
    image = plt.imread(file_path)
    dominant_color = extract_dominant_color(image)
    # Print the dominant color
    plt.imshow(image)
    plt.axis('off')
    print("Dominant color: " + str(dominant_color))
    plt.show()


load_image()
