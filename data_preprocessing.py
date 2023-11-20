import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


raw_data_dir = 'data\\raw_data\\'

# preprocessing polar coordinates data
polar_data_dir = raw_data_dir + 'polar'


for class_dir in os.listdir(polar_data_dir):
    class_dir = os.path.join(polar_data_dir, class_dir)
    for data_instance in os.listdir(class_dir):
        data_instance = os.path.join(class_dir, data_instance)
        print(data_instance)

        image = cv2.imread(data_instance)
        print(image.shape)
        if image.shape != (875, 1167, 3):
            exit('data is inconsistent size')

# Convert BGR to RGB (Matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.title('Polyacetal (POM) Image')
plt.axis('off')
# plt.show()