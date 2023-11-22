import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import *
from pylab import *
import random
from scipy.ndimage import map_coordinates, gaussian_filter
import json

# all spectra data is in the form of a 2d array
raw_data_dir = os.getcwd() + '\\data_processing\\raw_data\\'

# preprocessing spectra data
polar_data_dir = raw_data_dir + 'polar\\'

# Checking to make sure each example is the same dimensions
def check_data_size(data_dir):
    valid_dataset_size_check = 0
    invalid_dataset_size_check = 0
    plastic_obs = 0
    non_plastic_obs = 0
    image_1_path = data_dir + 'Polyacetal (POM)\\Polyacetal (POM)_90.png' 
    image_1 = cv2.imread(image_1_path)
    image_1_size = np.shape(image_1)

    class_dir_size_list = []

    for class_dir in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_dir)
        class_dir_size_list.append(len(os.listdir(class_dir)))
        if 'Cellulosic' in class_dir or 'Silica' in class_dir:
            non_plastic_obs += len(os.listdir(class_dir))
        else:
            plastic_obs += len(os.listdir(class_dir))
        for data_instance in os.listdir(class_dir):
            data_instance = os.path.join(class_dir, data_instance)
            data_instance = cv2.imread(data_instance)
            if np.shape(data_instance) == image_1_size:
                valid_dataset_size_check += 1
            else:
                invalid_dataset_size_check += 1
    return valid_dataset_size_check, invalid_dataset_size_check, class_dir_size_list, plastic_obs, non_plastic_obs


# Data augmentation method 1 - Horizontal shifting and horizontal flipping
def augment_strategy_1(raw_image):
    shift_value = random.randint(-270, 270)  # Change this to the number of pixels you want to shift the image
    augmented_image = np.roll(raw_image, shift=shift_value, axis=1)

    flip_chance = random.randint(0,1)
    if flip_chance == 0:
        augmented_image = np.flip(augmented_image, axis=1)

    return augmented_image



def elastic_transform_3d_color(image, alpha, sigma, random_state=None):
    """Elastic deformation of 3D color images."""
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    distorted_channels = []
    for channel in range(shape[2]):
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        distorted_channel = map_coordinates(image[:,:,channel], indices, order=1).reshape(shape[:2])
        distorted_channels.append(distorted_channel)

    distorted_image = np.stack(distorted_channels, axis=-1)

    return distorted_image

# # Load the image
testing_image_path = polar_data_dir + '\\Polyethylene (PE)\\Polyethylene (PE)_147.png' 
testing_image = cv2.imread(testing_image_path)
testing_image = cv2.cvtColor(testing_image, cv2.COLOR_BGR2RGB)

# # Apply elastic transformation with alpha (distortion factor) and sigma (smoothness)
#started a=50 s=5  --> a-(15,60) s->(2.5,4)
testing_image = elastic_transform_3d_color(testing_image, alpha=60, sigma=4)

# plt.imshow(testing_image)
# plt.title('Image')
# plt.axis('on')
# plt.show()
# import Image
testing_image_pil = Image.fromarray(testing_image)

# Resize the image
new_size = (1167, 876)
new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
box = ((new_size[0] - testing_image_pil.width) // 2, (new_size[1] - testing_image_pil.height) // 2)
new_im.paste(testing_image_pil, box)

plt.imshow(new_im)
plt.show()
# new_im.save('someimage.jpg')



# # Load the image
# testing_image_path = polar_data_dir + '\\Polyethylene (PE)\\Polyethylene (PE)_147.png' 
# testing_image = cv2.imread(testing_image_path)
# testing_image = cv2.cvtColor(testing_image, cv2.COLOR_BGR2RGB)

# # Add a white border to the image
# image_with_border = add_border(testing_image, border_thickness=20)

# # Show the images
# testing_image_path = polar_data_dir + '\\Polyethylene (PE)\\Polyethylene (PE)_147.png' 
# testing_image = np.array(cv2.imread(testing_image_path))
# testing_image = cv2.cvtColor(testing_image, cv2.COLOR_BGR2RGB)
# print(np.shape(testing_image), 'shape')

# plt.imshow(testing_image)
# plt.title(testing_image_path.split('\\')[-1])
# plt.axis('on')
# plt.show()






    # data statistics
# number_valid_examples, _, class_dir_size_list, plastic_obs, non_plastic_obs = check_data_size(polar_data_dir)
# print(f'Number of valid dataset samples: {number_valid_examples}')
# total_obs = plastic_obs + non_plastic_obs
# print(f'Num plastic obs: {plastic_obs}, Num non-plastic obs: {non_plastic_obs}, total obs: {total_obs}')
# largest_class = max(class_dir_size_list)
# print(f'Largest class size: Silica, {largest_class}')


