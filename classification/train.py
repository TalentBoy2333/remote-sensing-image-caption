import numpy as np 
import torch
import torch.utils.data as Data
from dataset import RESISC45 
from classifier import Classifier
from augmentations import Augmentation

cuda = True if torch.cuda.is_available() else False

def train(model_name='resnet', batch_size=64, train_epoch=10, learning_rate=1e-4, load_model=None):
    """
    Train.
    :param model_name: pre-train model. a str
    :param batch_size: batch size. a int
    :param train_epoch: How many epoch. a int
    :param learning_rate: learning rate. a float
    :param load_model: the parameters of classifier model which has been trained. a str(folder)
    """
    dataset = RESISC45('./', Augmentation())
    class_number = len(dataset.classes)
    classifier = Classifier(model_name, class_number, True)
    if cuda:
        classifier = classifier.cuda()
    if load_model:
        classifier.load_state_dict(torch.load(load_model))
        
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    data_loader = Data.DataLoader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True)
    # batch_iterator = iter(data_loader)
    for epoch in range(train_epoch):
        for t, (batch_image, batch_label) in enumerate(data_loader):
            # print('batch_image:', batch_image.shape)
            # print('batch_label:', batch_label.shape)
            if cuda:
                batch_image = batch_image.cuda()
                batch_label = batch_label.cuda()
            prediction = classifier(batch_image)
            # print(prediction.size())
            class_predict = np.argmax(prediction.cpu().data.numpy(), axis=1)
            same = 0
            for y_pred, y in zip(class_predict, batch_label.cpu().data.numpy()):
                if y_pred == y:
                    same += 1
            accuracy = same / batch_image.shape[0]

            loss = loss_func(prediction, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Epoch', epoch+1,
                '| Iter', t, 
                '| loss:', loss.cpu().data.numpy(),
                '| accuracy:', accuracy, 
                '| batch size:', batch_size, 
                '| learning rate:', learning_rate, 
            )
        print('Saveing classifier model for epoch', epoch+1, '.')
        torch.save(classifier.state_dict(), './models/train/'+model_name+'_'+str(epoch+1)+'.pkl')






if __name__ == '__main__':
    train('mobilenet', 100, 100, 1e-4)