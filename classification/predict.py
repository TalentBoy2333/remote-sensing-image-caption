import numpy as np 
import torch
import cv2
from dataset import RESISC45 
from classifier import Classifier
import matplotlib.pyplot as plt
import os

cuda = True if torch.cuda.is_available() else False
dataset = RESISC45('./classification')
classes = dataset.classes

def get_classifier(model_name='resnet', load_model=None):
    classifier = Classifier(model_name, len(classes))
    classifier.eval()
    if cuda:
        classifier = classifier.cuda()
    if load_model:
        print('Loading parameters of model.')
        classifier.load_state_dict(torch.load(load_model))
    return classifier

def predict(classifier, image_path, display=False):
    image = cv2.imread(image_path)
    # image_transform = cv2.resize(image, (256, 256))
    image_transform = image.astype(np.float32) / 255.0
    image_transform = image_transform.transpose([2,0,1]) 
    image_transform = np.expand_dims(image_transform, axis=0)
    image_transform = torch.from_numpy(image_transform).type(torch.FloatTensor)
    if cuda:
        image_transform = image_transform.cuda()
    prediction = classifier(image_transform)
    class_predict = np.argmax(prediction.cpu().data.numpy())
    class_name = classes[class_predict]
    if display:
        plt.figure()
        plt.imshow(image)
        plt.title(class_name)
        plt.xticks(())
        plt.yticks(())
        plt.show()
    return class_name


if __name__ == '__main__':
    image_name = 'GF1_PMS2_E121.4_N25.2_20141207_L1A0000502922-PAN2.jpg'
    classifier = get_classifier('resnet', './classification/models/train/classifier_50.pkl')
    predict(classifier, image_name, True)