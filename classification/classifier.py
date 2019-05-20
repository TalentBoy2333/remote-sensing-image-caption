import numpy as np 
import torch 
from torch.autograd import Variable
# from classification.resnet import resnet101
from resnet import resnet101
from MobileNetV2 import MobileNetV2

class Classifier(torch.nn.Module):
    """
    Classifier
    """
    def __init__(self, model_name='resnet', classes_number=20, is_train=False):
        """
        :param model_name: pre-train model name. a str
        :param classes_number: number of classes. a int
        """
        super(Classifier, self).__init__()
        self.model_name = model_name
        self.is_train = is_train
        if model_name == 'resnet':
            self.feature_net = resnet101(pretrained=is_train)
        elif model_name == 'mobilenet':
            net = MobileNetV2(n_class=1000)
            if is_train:
                state_dict = torch.load('./models/mobilenet_v2.pth.tar') # add map_location='cpu' if no gpu
                net.load_state_dict(state_dict)
            self.feature_net = net.features
        else:
            print('Error: wrong model name.')
            exit()

        if model_name == 'mobilenet':
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.drop_out = torch.nn.Dropout(0.5)
            self.fc = torch.nn.Linear(1280, classes_number)
        else:
            self.drop_out = torch.nn.Dropout(0.5)
            self.fc = torch.nn.Linear(2048, classes_number)

        self.fc.weight.data.normal_(0, 0.1)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images matrix. a tensor size (batch_size, 3, 256, 256)
        :return: logist of each class. a tensor size (batch_size, cls_num)
        """
        feature = self.feature_net(images)
        if self.model_name == 'mobilenet':
            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
        # print('feature:', feature.size())
        if self.is_train:
            feature = self.drop_out(feature)
        output = self.fc(feature)
        return output




if __name__ == '__main__':
    classifier = Classifier('mobilenet')
    print(classifier)

    fake_images = np.random.rand(5,3,224,224)
    fake_images = torch.from_numpy(fake_images)
    fake_images = Variable(fake_images).type(torch.FloatTensor)
    print('fake_images:', fake_images.size())

    output = classifier(fake_images)
    print('output:', output.size())