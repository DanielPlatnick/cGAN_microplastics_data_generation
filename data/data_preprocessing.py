import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

raw_data_dir = os.getcwd() + '\\data\\raw_data\\'

# preprocessing spectra data
spectra_data_dir = raw_data_dir + 'spectra\\'

class_dir_size_list = []

# Checking to make sure each example is the same dimensions
def check_data_size(data_dir):
    valid_dataset_size_check = 0
    invalid_dataset_size_check = 0
    plastic_obs = 0
    non_plastic_obs = 0
    image_1_path = data_dir + 'Polyacetal (POM)\\Polyacetal (POM)_90.png' 
    image_1 = cv2.imread(image_1_path)
    image_1_size = np.shape(image_1)

    for class_dir in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_dir)
        class_dir_size_list.append(len(os.listdir(class_dir)))
        if 'Cellulosic' in class_dir or 'Silica' in class_dir:
            non_plastic_obs += len(os.listdir(class_dir))
        else:
            plastic_obs += len(os.listdir(class_dir))
        # print(class_dir, len(os.listdir(class_dir)))
        for data_instance in os.listdir(class_dir):
            data_instance = os.path.join(class_dir, data_instance)
            data_instance = cv2.imread(data_instance)
            if np.shape(data_instance) == image_1_size:
                # print(type(np.shape(data_instance)))
                # print(np.shape(data_instance))
                valid_dataset_size_check += 1
            else:
                invalid_dataset_size_check += 1
    return valid_dataset_size_check, invalid_dataset_size_check, class_dir_size_list, plastic_obs, non_plastic_obs
        # print(data_instance)


number_valid_examples, number_invalid_examples, class_dir_size_list, plastic_obs, non_plastic_obs = check_data_size(spectra_data_dir)
print(f'Number of valid dataset samples: {number_valid_examples}')
print(f'Number of invalid dataset samples: {number_invalid_examples}')
total_obs = plastic_obs + non_plastic_obs
print(f'Num plastic obs: {plastic_obs}, Num non-plastic obs: {non_plastic_obs}, total obs: {total_obs}')

largest_class = max(class_dir_size_list)
print(f'Largest class size: Silica, {largest_class}')



spectral_data = []

image_1_path = spectra_data_dir + 'Polyacetal (POM)\\Polyacetal (POM)_90.png' 
image_1 = cv2.imread(image_1_path)
image_1_size = np.shape(image_1)

image_1_array = np.array(image_1)
flattened_array = image_1_array[:, :, 0]



plt.imshow(flattened_array, cmap='gray')
# plt.imshow(image_1_array, cmap='gray')
plt.title(image_1_path.split('\\')[-1])
plt.axis('off')
# plt.show()


# test_data_instance = 
 



# print(image_1_array, np.shape(image_1_array))


# for i in range(image_1_array.shape[0]):
#     for j in range(image_1_array.shape[1]):
#         if not np.all(image_1_array[i, j] == [255, 255, 255]):
#             print(f"Pixel at position ({i}, {j}): {image_1_array[i, j]}")

# plt.imshow(image)
# plt.title(data_instance.split('\\')[-1])
# plt.axis('on')
# plt.show()


# Random variations in offset, multiplication, and slope











# polar_data_dir = raw_data_dir + 'polar'


# for class_dir in os.listdir(polar_data_dir):
#     class_dir = os.path.join(polar_data_dir, class_dir)
#     for data_instance in os.listdir(class_dir):
#         data_instance = os.path.join(class_dir, data_instance)
#         print(data_instance)

# plastic_obs = 0
# non_plastic_obs = 0
# for class_dir in os.listdir(polar_data_dir):
#     class_dir = os.path.join(polar_data_dir, class_dir)
#     if 'Cellulosic' in class_dir or 'Silica' in class_dir:
#         non_plastic_obs += len(os.listdir(class_dir))
#     else:
#         plastic_obs += len(os.listdir(class_dir))

        
# total_obs = plastic_obs + non_plastic_obs
# print(plastic_obs, non_plastic_obs, total_obs)

# image = cv2.imread(data_instance)
# # # print(image.shape)
# # if image.shape != (875, 1167, 3):
# #     exit('data is inconsistent size')


# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# plt.imshow(image_rgb)
# plt.title('Polyacetal (POM) Image')
# plt.axis('off')
# # plt.show()
