import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import AppLayout, HBox, VBox, Output
import nibabel as nib
import os
from scipy.ndimage import zoom
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sys

def crop_image(input_array,target_shape):
    start_x = (input_array.shape[1] - target_shape[1]) // 2
    start_y = (input_array.shape[2] - target_shape[2]) // 2
    start_z = (input_array.shape[0] - target_shape[0]) // 2

    cropped_array = input_array[start_z:start_z+target_shape[0], start_x:start_x+target_shape[1], start_y:start_y+target_shape[2]]

    return cropped_array

def kill_background(model_array):
    nonzero_indices = np.nonzero(model_array)

    # 找到前景部分的最小和最大索引值
    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)
    #print(max_indices)
    # 计算最小边界框的切片范围
    slices = [slice(min_index, max_index + 1) for min_index, max_index in zip(min_indices, max_indices)]

    # 根据切片范围裁剪出包含3D模型的最小numpy数组
    min_cropped_array = model_array[tuple(slices)]

    return min_cropped_array

def prepare_single_image(image_path):
    good_image = nib.load(image_path)
    good_data = good_image.get_fdata()
    array = np.array(good_data)

    array = np.transpose(array, (2, 0, 1))

    array = kill_background(array)

    #array = zoom(array, (20/array.shape[0], 160/array.shape[1], 160/array.shape[2]))
    
    max_value = np.max(array)
    min_value = np.min(array)
    normalized_array = (array - min_value) / (max_value - min_value)

    normalized_array = np.clip(normalized_array, 0, 1)
    
    return normalized_array

def prepare_single_image_for_urru(image_path):
    good_image = nib.load(image_path)
    good_data = good_image.get_fdata()
    array = np.array(good_data)

    array[array<500] = 0
    
    array = np.transpose(array, (2, 0, 1))

    array = kill_background(array)

    #array = zoom(array, (20/array.shape[0], 160/array.shape[1], 160/array.shape[2]))
    
    max_value = np.max(array)
    min_value = np.min(array)
    normalized_array = (array - min_value) / (max_value - min_value)

    normalized_array = np.clip(normalized_array, 0, 1)
    
    return normalized_array

def prepare_image_for_test(image_path):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160), antialias=True),
    ])
    
    good_image = nib.load(image_path)
    good_data = good_image.get_fdata()
    array = np.array(good_data)

    array = np.transpose(array, (2, 0, 1))

    array = kill_background(array)

    #array = zoom(array, (20/array.shape[0], 160/array.shape[1], 160/array.shape[2]))
    
    max_value = np.max(array)
    min_value = np.min(array)
    normalized_array = (array - min_value) / (max_value - min_value)

    normalized_array = np.clip(normalized_array, 0, 1)
    
    image = np.expand_dims(normalized_array, axis=0)
    image = image[0, :, :, :]

    middle_index = image.shape[0] // 2
    start_index = max(0, middle_index - 3)
    end_index = min(image.shape[0], middle_index + 3)

    selected_slices = image[start_index:end_index+1, :, :]

    images = []  # 存储五张图像
    for i in range(selected_slices.shape[0]):
        slice = selected_slices[i:i+1, :, :]
        images.append(slice)


    transformed_imgs = []
    for img in images:
        img = img.transpose(1, 2, 0)
        img = trans(img)
        transformed_imgs.append(img)
    
    imgs = torch.cat(transformed_imgs, dim=0)

    return imgs

def change_and_save_image(input_path,output_path):
    good_path = input_path
    good_list = os.listdir(good_path)
    good_list = sorted(good_list)

    i = 0
    for good in good_list:
        image_path = good_path+'/'+good

        try:
            array = prepare_single_image(image_path)

            #array = zoom(array, [(20/array.shape[0]),1,1], order=3)  # order=1 表示使用线性插值

            #print(array.shape)

            np.save(output_path+'/'+str(i)+'_'+good+'.npy', array)
            print(image_path+' save to '+'output_path'+'/'+str(i)+'.npy')
            i+=1
        except Exception:
            print(image_path+' data die!')
            pass 
        
        #if i == 120:
            #break
    print('complete!')

# Function to process images and return file names and arrays

def change_and_save_image_inference(input_path):
    print('change_and_save_image_inference start')
    good_path = input_path
    good_list = os.listdir(good_path)
    good_list = sorted(good_list)

    file_names = []  # List to store file names
    processed_arrays = []  # List to store processed arrays

    for i, good in enumerate(good_list):
        image_path = good_path + '/' + good

        try:
            array = prepare_single_image(image_path)

            # Save the processed array and file name
            processed_arrays.append(array)
            file_names.append( good)
            print(image_path + ' processed and filename returned: ' + file_names[-1])
        
        except Exception:
            print(image_path + ' data processing failed!')
            pass 
    
    print('Processing complete!')
    return file_names, processed_arrays

def extract_content_between_last_two_slashes(path):
    """
    Extracts and returns the content between the last two slashes in the given path string.
    
    Parameters:
    path (str): The path string from which to extract content.
    
    Returns:
    str: The content between the last two slashes, or an empty string if not enough slashes are present.
    """
    # 找到最后一个 '/' 的位置
    last_slash_index = path.rfind('/')
    
    # 如果没有找到 '/', 返回空字符串
    if last_slash_index == -1:
        return ""
    
    # 找到倒数第二个 '/' 的位置
    second_last_slash_index = path.rfind('/', 0, last_slash_index)
    
    # 如果没有找到第二个 '/', 返回空字符串
    if second_last_slash_index == -1:
        return ""
    
    # 提取最后两个 '/' 之间的内容
    content_between_slashes = path[second_last_slash_index + 1:last_slash_index]
    
    return content_between_slashes

def change_and_save_fidon_atlas(input_path_list,output_path):
    good_list = input_path_list

    i = 0
    for good in good_list:
        image_path = good

        array = prepare_single_image(image_path)

        #array = zoom(array, [(20/array.shape[0]),1,1], order=3)  # order=1 表示使用线性插值

        #print(array.shape)
        good = extract_content_between_last_two_slashes(good)
        np.save(output_path+'/'+str(i)+'_'+good+'.npy', array)
        print(image_path+' save to '+'output_path'+'/'+str(i)+'.npy')
        i+=1
        
        #if i == 120:
            #break
    print('complete!')

def extract_after_last_slash(input_string):
    """
    Extracts and returns the substring that appears after the last slash ('/') in the input string.
    
    Parameters:
    input_string (str): The string from which to extract the substring.
    
    Returns:
    str: The substring after the last slash, or an empty string if no slash is present.
    """
    # 查找最后一个 '/' 的位置
    last_slash_index = input_string.rfind('/')
    
    # 如果找到了 '/', 提取其后的内容
    if last_slash_index != -1:
        return input_string[last_slash_index + 1:]
    else:
        # 如果没有 '/', 返回空字符串
        return ""
def change_and_save_gholipour_atlas(input_path_list,output_path):
    good_list = input_path_list

    i = 0
    for good in good_list:
        image_path = good

        if 'urru' in output_path:
            array = prepare_single_image_for_urru(image_path)
        else:
            array = prepare_single_image(image_path)
        
        good = extract_after_last_slash(good)
        
        np.save(output_path+'/'+str(i)+'_'+good+'.npy', array)
        print(image_path+' save to '+'output_path'+'/'+str(i)+'.npy')
        i+=1
        
        #if i == 120:
            #break
    print('complete!')



class MRIDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.data = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    file_path = os.path.join(class_dir, filename)
                    self.data.append(file_path)
                    self.labels.append(class_idx)
        
        self.data, self.labels = np.array(self.data), np.array(self.labels)
        
        if self.mode == 'train':
            self.data, _, self.labels, _ = train_test_split(self.data, self.labels, test_size=0.99, random_state=42)
        elif self.mode == 'test':
            _, self.data, _, self.labels = train_test_split(self.data, self.labels, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                # 计算0像素的比例
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

                
        elif self.transform and self.mode=='test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)

            selected_slices = image[start_index:end_index+1, :, :]

            images = []  # 存储五张图像
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)


            image = self.transform(images)
            
        return image, label
    

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class AtlasDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_names = ['cor', 'tra', 'sag']
        self.data = []
        self.labels = []

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('L')
        img = transforms.ToTensor()(img)
        img = transforms.Resize((160, 160), antialias=True)(img)
        angle = random.randint(0, 360)
        img = transforms.functional.affine(img, angle, translate=(0, 0), scale=1, shear=0)

        # 随机水平翻转，概率为 0.5
        img = transforms.RandomHorizontalFlip(p=0.5)(img)

        # 随机垂直翻转，概率为 0.5
        img = transforms.RandomVerticalFlip(p=0.5)(img)

        return img, label


import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MRIDataset_quality(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self.mode == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                # 计算0像素的比例
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

                
        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)

            selected_slices = image[start_index:end_index+1, :, :]

            images = []  # 存储五张图像
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            image = self.transform(images)
            
        return image, label_ori, label_quality

import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MRIDataset_quality_new(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            # 计算每个切片的非0区域占比
            non_zero_ratios = [np.mean(slice != 0) for slice in image]

            # 获取非0区域占比最大的七个切片的索引
            top_slices_indices = np.argsort(non_zero_ratios)[-7:]

            # 选择这些切片
            selected_slices = image[top_slices_indices, :, :]

            images = []  # 存储七张图像
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            image = self.transform(images)
            
        return image, label_ori, label_quality


class MRIDataset_quality_new2(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            # 计算每个切片的非0区域占比
            non_zero_ratios = [np.mean(slice != 0) for slice in image]

            # 获取非0区域占比最大的七个切片的索引
            top_slices_indices = np.argsort(non_zero_ratios)[-7:]

            # 选择这些切片
            selected_slices = image[top_slices_indices, :, :]

            images = []  # 存储七张图像
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            #print(image.shape)
            image = self.transform(images)
            
        return image, label_ori, label_quality, image_path



class MRIDataset_quality_new2_slice5(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            # 计算每个切片的非0区域占比
            non_zero_ratios = [np.mean(slice != 0) for slice in image]

            # 获取非0区域占比最大的七个切片的索引
            top_slices_indices = np.argsort(non_zero_ratios)[-5:]

            # 选择这些切片
            selected_slices = image[top_slices_indices, :, :]

            images = []  # 存储七张图像
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            image = self.transform(images)
            
        return image, label_ori, label_quality, image_path

class MRIDataset_quality_new2_slice9(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]
            non_zero_ratios = [np.mean(slice != 0) for slice in image]
            top_slices_indices = np.argsort(non_zero_ratios)[-9:]

            selected_slices = image[top_slices_indices, :, :]

            images = [] 
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            image = self.transform(images)
            
        return image, label_ori, label_quality, image_path


class MRIDataset_quality_resnet18(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.class_names = ['cor', 'tra', 'sag']
        self.quality_names = ['good', 'bad']
        self.data = []
        self.labels_ori = []
        self.labels_quality = []
        
        for class_idx, class_name in enumerate(self.class_names):
            for quality_idx, quality_name in enumerate(self.quality_names):
                quality_dir = os.path.join(root_dir, class_name, quality_name)
                for filename in os.listdir(quality_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(quality_dir, filename)
                        self.data.append(file_path)
                        self.labels_ori.append(class_idx)
                        self.labels_quality.append(quality_idx)
        
        self.data, self.labels_ori, self.labels_quality = np.array(self.data), np.array(self.labels_ori), np.array(self.labels_quality)
        
        if self.mode == 'train':
            self.data, _, self.labels_ori, _, self.labels_quality, _ = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        elif self == 'test':
            _, self.data, _, self.labels_ori, _, self.labels_quality = train_test_split(
                self.data, self.labels_ori, self.labels_quality, test_size=0.99, random_state=42)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_ori = self.labels_ori[idx]
        label_quality = self.labels_quality[idx]
        
        image = np.load(image_path)
        
        if self.transform and self.mode == 'train':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            middle_index = image.shape[0] // 2
            start_index = max(0, middle_index - 3)
            end_index = min(image.shape[0], middle_index + 3)
            
            max_iterations = 5
            iterations = 0
            
            while iterations < max_iterations:
                random_slice_index = np.random.randint(start_index, end_index)
                selected_slice = image[random_slice_index:random_slice_index + 1, :, :]
                
                zero_ratio = np.mean(selected_slice == 0)
                
                if zero_ratio <= 0.5:
                    break
                
                iterations += 1
            
            image = selected_slice

            image = self.transform(image)

        elif self.transform and self.mode == 'test':
            image = np.expand_dims(image, axis=0)
            image = image[0, :, :, :]

            non_zero_ratios = [np.mean(slice != 0) for slice in image]

            top_slices_indices = np.argsort(non_zero_ratios)[-14:]

            selected_slices = image[top_slices_indices, :, :]

            images = [] 
            for i in range(selected_slices.shape[0]):
                slice = selected_slices[i:i+1, :, :]
                images.append(slice)

            image = self.transform(images)
            
        return image, label_ori, label_quality, image_path


import numpy as np

def traverse_and_modify(image_list, image_name_list):
    updated_image_list = []
    updated_name_list = []

    for data, file_name in zip(image_list, image_name_list):
        try:
            if len(data.shape) != 3:
                print(f"Skipping {file_name}: Not a 3D data")
                continue
            
            num_slices = data.shape[0]
            if num_slices < 7:
                print(f"Deleted {file_name} because its first dimension size is {num_slices}")
                continue
            elif 7 <= num_slices < 14:
                non_zero_counts = [(i, np.count_nonzero(data[i])) for i in range(num_slices)]
                sorted_slices = sorted(non_zero_counts, key=lambda x: x[1], reverse=True)
                
                slices_to_expand = [idx for idx, _ in sorted_slices[:14 - num_slices]]
                for idx in sorted(slices_to_expand):
                    data = np.insert(data, idx + 1, data[idx], axis=0)
                
                print(f"Expanded {file_name} to 14 slices")
            elif num_slices > 14:
                non_zero_counts = [(i, np.count_nonzero(data[i])) for i in range(num_slices)]
                sorted_slices = sorted(non_zero_counts, key=lambda x: x[1])
                
                slices_to_remove = [idx for idx, _ in sorted_slices[:num_slices - 14]]
                for idx in sorted(slices_to_remove, reverse=True):
                    data = np.delete(data, idx, axis=0)
                
                print(f"Reduced {file_name} to 14 slices")

            updated_image_list.append(data)
            updated_name_list.append(file_name)

        except Exception as e:
            print(f"Could not process {file_name}: {e}")

    return updated_image_list, updated_name_list



import numpy as np
import torch
import nibabel as nb
import pandas as pd
from skimage.transform import resize
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from dataloader import *
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch


seed = 142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
batch_size = 1
      
import torchio as tio

class testTransform(object):
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160), antialias=True),
        ])

    def __call__(self, img_list):
        transformed_imgs = []
        for img in img_list:
            #print(img.shape)
            img = img.transpose(1, 2, 0)
            img = self.transforms(img)
            transformed_imgs.append(img)
        
        imgs = torch.cat(transformed_imgs, dim=0)
        return imgs

data_transform_test=  transforms.Compose([
    testTransform(),
])


import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import sys
slice_used = 14
class MRIDatasetQuality_inference(Dataset):
    def __init__(self, image_list, image_name_list, transform=None):
        self.image_list = image_list
        self.image_name_list = image_name_list
        self.transform = transform
        self.image_list = image_list
        self.image_name_list = image_name_list 

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        image_path = self.image_name_list[idx]

        image = np.expand_dims(image, axis=0) 
        image = image[0, :, :, :] 

        non_zero_ratios = [np.mean(slice != 0) for slice in image]
        top_slices_indices = np.argsort(non_zero_ratios)[-slice_used:]

        selected_slices = image[top_slices_indices, :, :]

        images = []
        for slice in selected_slices:
            images.append(slice[np.newaxis, :, :])

        if self.transform:
            image = self.transform(images)

        return image, image_path
