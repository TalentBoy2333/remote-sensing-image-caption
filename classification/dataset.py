import torch.utils.data as Data
import torch
import os
import cv2 
import numpy as np 
# from classification.augmentations import Augmentation
from augmentations import Augmentation

class RESISC45(Data.Dataset):
    """
    Dataset RESISC45
    """
    def __init__(self, root, transform=None):
        """
        :param root: the path of this project. a str
        :param transform: augmentation for images. 
        """
        self.root = root 
        self.transform = transform
        image_path = os.path.join(root, 'data', 'NWPU-RESISC45')
        train_path = os.path.join(root, 'data', 'train')
        self.train_path = train_path
        have_save = os.listdir(train_path)
        if have_save:
            load_name = os.path.join(train_path, 'classes.npy')
            self.classes = np.load(load_name)
            load_name = os.path.join(train_path, 'image_label.npy')
            self.image_label = np.load(load_name)
            self.image_number = len(self.image_label)
        else:
            self.classes = []
            self.image_label = []
            self.save_all_image(image_path, train_path)

        print('Dataset size:', self.image_number)
        print('Classes:', self.classes)    
        

    def __getitem__(self, index):
        """
        torch.utils.data.DataLoader can find this function and get data.
        :param index: the index of image. a int
        :return: image, a tensor size(3,256,256) and class, a tensor size(1,)
        """
        image, gt = self.pull_item(index)
        return image, gt

    def __len__(self):
        """
        This function can: 
        dataset = RESISC45('./')
        epoch_size = len(dataset)
        """
        return self.image_number

    def save_all_image(self, image_path='./data/NWPU-RESISC45', save_path='./data/train'):
        """
        Image in many path of image_path ==> Image in save_path
        :param image_path: Dataset path. a str
        :param save_path: train data path. a str
        """
        self.image_number = 0
        self.class_number = 0
        classes = os.listdir(image_path)
        for c in classes:
            self.classes.append(c)
            print('Saving images of class', c)
            # print(self.class_number)
            path = os.path.join(image_path, c)
            images = os.listdir(path)
            for name in images:
                image_name = os.path.join(path, name)
                image = cv2.imread(image_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                save_name = str(self.image_number).zfill(6)
                save_name = os.path.join(save_path, save_name+'.jpg')
                cv2.imwrite(save_name, image)
                self.image_label.append(self.class_number)
                self.image_number += 1
            self.class_number += 1

        save_name = os.path.join(save_path, 'classes.npy')
        np.save(save_name, self.classes)
        save_name = os.path.join(save_path, 'image_label.npy')
        np.save(save_name, self.image_label)

    def pull_item(self, index):
        """
        Get one data of dataset.
        :param index: the index of image. a int
        :return: image, a tensor size(1,3,256,256), and label, a tensor size(1,1)
        """
        train_path = self.train_path
        image_name = str(index).zfill(6) + '.jpg'
        image_name = os.path.join(train_path, image_name)
        image = cv2.imread(image_name)
        if self.transform is not None:
            image = self.transform(image)
        image = image / 255.0
        image = image.transpose([2,0,1]) 
        image = torch.from_numpy(image).type(torch.FloatTensor)

        label = np.array(self.image_label[index])
        label = torch.from_numpy(label).type(torch.LongTensor)
        return image, label

    



if __name__ == '__main__':
    dataset = RESISC45('./', Augmentation())
    epoch_size = len(dataset)
    print('epoch_size:', epoch_size)
    data_loader = Data.DataLoader(dataset, 32, num_workers=0, shuffle=True, pin_memory=True)
    batch_iterator = iter(data_loader)
    images, labels = next(batch_iterator)
    print('images:', images.shape)
    print('labels:', labels.shape)
    # for i in range(1000):
    #     images, labels = next(batch_iterator)
    #     print('Iter', i)
    #     print('images:', images.shape)
    #     print('labels:', labels.shape)
