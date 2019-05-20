from config import Config 
import json
import os

cfg = Config()

class Data():
    """
    Data.
    """
    def __init__(self):
        self.image_number = 0
        self.sentence_length = 0 # the max length of all sentences.
        self.images_list = [] # save image name
        self.annotations_list = [] # save five sentence for every image.
        """
        # '<start>': start flag
        # ' ': padding for vary length sentences
        # '.': end flag
        """
        self.dictionary = [' ', '<start>', '.']
        self.train_list = [] # image index for train
        self.val_list = [] # image index for validation
        self.test_list = [] # image index for test

    def get_images_list(self):
        """
        Get names of all images to a list.
        """
        self.images_list = os.listdir(cfg.images_folder)
        self.image_number = len(self.images_list)
        # Create annotations_list for all images.
        for _ in self.images_list:
            self.annotations_list.append([])
        print(
            'Images path:', cfg.images_folder, 
            '\nGet', self.image_number, 'images.', 
            '\nAnnotations length:', len(self.annotations_list)
        )
    
    def get_annotations(self):
        """
        Get five sentences of every image in the '.json' file.
        """
        file = open(cfg.annotations_name, encoding='utf-8')
        load_dict = json.load(file)
        # The file has two element: dataset, and images, we take images.
        for element in load_dict:
            print(cfg.annotations_name, 'has element', '[', element, ']')

        print('Getting annotations for [ images ]')
        for image in load_dict['images']:
            """
            # 'images' element has many key.
            # key filename: save the name of image's.
            # key sentences: save five captions for the image(filename)
            # key sentences['raw']: five str.
            # key sentences['tokens']: five list.
            """
            # print(image['split'])
            filename = image['filename']
            if filename not in self.images_list:
                continue
            index = self.images_list.index(filename)
            if image['split'] == 'train':
                self.train_list.append(index)
            elif image['split'] == 'val':
                self.val_list.append(index)
            else:
                self.test_list.append(index)
            for sentence in image['sentences']:
                # print(sentence)
                # print(sentence['raw'])
                # print(sentence['tokens'])
                """
                # In sentences['raw'], a word may be connected by a ','.
                # For example, 'I am fine, and you.', 'fine,' is a word.
                # But, we want divide 'fine,' to two words: ['fine', ',']
                # So, we add a ' ' into 'fine' and ','.
                """
                sentence_temp = ''
                for word in sentence['raw']:
                    if ',' in word:
                        word = ' ' + ','
                    sentence_temp += word 
                self.annotations_list[index].append(sentence_temp)
                element = sentence_temp.split(' ')
                # push new word into dictionary.
                for word in element:
                    if word not in self.dictionary:
                        self.dictionary.append(word)
                # update the max length of all sentences.
                if len(element) > self.sentence_length:
                    self.sentence_length = len(element)
        print('Sentence length:', self.sentence_length)
        print('Dictionary length:', len(self.dictionary))
        print('The number of train data: ', len(self.train_list))
        print('The number of validation: ', len(self.val_list))
        print('The number of test data: ', len(self.test_list))

    def text_all_images(self):
        """
        Write all sentences into '.txt' files.
        """
        print('Writing texts for images.')
        for image_index in range(self.image_number):
            filename = self.images_list[image_index]
            sentences = self.annotations_list[image_index]
            text = open('./data/texts/'+filename[:-4]+'.txt', 'w')
            for sentence in sentences:
                text.write(sentence)
                text.write('\n')
            text.close()


if __name__ == '__main__':
    data = Data()
    data.get_images_list()
    data.get_annotations()
    # data.text_all_images()

    print(data.dictionary[0:10])
    print(data.images_list[0])
    print(data.annotations_list[0])

    
