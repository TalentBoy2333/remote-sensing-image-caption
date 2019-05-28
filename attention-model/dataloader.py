from config import Config 
import numpy as np
import matplotlib.pyplot as plt
import os
from data import Data
from augmentations import Augmentation

cfg = Config()

class DataLoader():
    """
    Data Loader.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.data = Data()
        self.init_batch()

    def init_batch(self):
        """
        Initial the data loader, calculate sample number, reset image_index, 
        and shuffle the name list.
        """
        self.image_index = 0
        self.image_index_list = self.data.train_list
        self.sample_number = len(self.image_index_list) * 5
        # print(len(image_index_list))
        np.random.shuffle(self.image_index_list)
        # print(self.image_index_list)

    def get_one_data(self, index, show=False):
        """
        Get one image, and one caption of the image.
        :param index: the index of a image. a int number 
        :param show: show the image and sentence or not. True or False
        :return image: the image matrix. a np.array size (1, 3, 224, 224)
        :return label: the sentence label of the image. a np.array size (1, 35)
        """
        ind_list = self.image_index_list
        img_num = len(ind_list)
        images = self.data.images
        sen_len = self.data.sentence_length
        anno_list = self.data.annotations
        dic = self.data.dictionary
        """
        # We have [img_num] images, but [img_num * 5] sentences for [img_num] images.
        # So, the real image index is (index%img_num).
        # For example, we got index 40956, but we haven't 40956th image, 
        #              the real index is 40956 % 10000 = 956
        # Go on, the 956th image has five sentences, so which should we take?
        # We calculate the index of label by int(index/10000)
        #              the index of the image is 40956, 
        #              int(40956 / 10000) = 4
        # So, we should take the 4th(we can take 0th, 1st, 2nd, 3rd, 4th) sentence of this image.
        """
        label_index = int(index / img_num)
        real_index = index % img_num
        image = images[ind_list[real_index]]
        if self.transform is not None:
            image = self.transform(image)
        if show:
            image_ = image[:,:,::-1].astype(np.uint8)
        image = image / 255.0
        image = image.transpose([2,0,1]) 
        image = np.expand_dims(image, axis=0)
                
        label = np.zeros(sen_len)
        sentence = anno_list[ind_list[real_index]][label_index]
        element = sentence.split(' ')
        for i, word in enumerate(element):
            label[i] = dic.index(word)
        label = np.expand_dims(label, axis=0)

        if show:
            plt.figure(1)
            plt.imshow(image_, interpolation='nearest', origin='upper')
            plt.title(sentence)
            plt.xticks(())
            plt.yticks(())
            plt.show()
        return image, label

    def get_next_batch(self):
        """
        Get a batch of data(images, labels).
        :return batch_image: a batch of images. a np.array size(batch_size, 3, 224, 224)
        :return batch_label: a batch of labels. a np.array size(batch_size, 35)
        """
        img_num = self.sample_number
        print('Getting a batch of data.')
        for i in range(cfg.batch_size):
            image, label = self.get_one_data(self.image_index)
            if i == 0:
                batch_image = image
                batch_label = label
            else:
                batch_image = np.concatenate([batch_image, image], axis=0)
                batch_label = np.concatenate([batch_label, label], axis=0)

            self.image_index += 1
            if self.image_index == img_num:
                self.init_batch()
        """
        # For example, we get a batch of label, batch size is 5.
            label: (1, 35)
            [[  3. 713. 221.  33.   5.  51.  52.  34.  35.  12.  13. 387. 100.   2.
                0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.]
            [ 15.  34.  35.   5.  12. 170.  86.  20.   9. 112.  27.   9. 962. 470.
            62.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.]
            [ 49.  18.   9. 171.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.]
            [562.  20.  16. 837.  18. 268. 397.   2.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.]
            [  9. 647.  69.  34. 544. 318. 434. 435.  27.  32. 114. 698.  20.  38.
            42.  12.  62.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.   0.   0.   0.]]
        # We don't need these zeros in the end of label, so we delete them, align.
        # And got a new label: (1, 18)
            [[  3. 713. 221.  33.   5.  51.  52.  34.  35.  12.  13. 387. 100.   2.
                0.   0.   0.   0.]
            [ 15.  34.  35.   5.  12. 170.  86.  20.   9. 112.  27.   9. 962. 470.
                62.   2.   0.   0.]
            [ 49.  18.   9. 171.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.
                0.   0.   0.   0.]
            [  9. 647.  69.  34. 544. 318. 434. 435.  27.  32. 114. 698.  20.  38.
                42.  12.  62.   2.]]
        """
        for i in np.arange(35)[::-1]:
            if batch_label[:,i].any():
                break
        batch_label = batch_label[:, :i+1]
        return batch_image, batch_label




if __name__ == '__main__':
    dataloader = DataLoader(Augmentation())
    image, label = dataloader.get_one_data(2450, True)
    print('image:', image.shape)
    print('label:', label)

    # batch_image, batch_label = dataloader.get_next_batch()
    # print(batch_image.shape)
    # print(batch_label.shape)
    # print(batch_label)
    