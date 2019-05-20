import numpy as np 
import torch 
from torch.autograd import Variable
from config import Config
from model import Encoder, DecoderWithAttention
from data import Data 
import cv2
import matplotlib.pyplot as plt 

cuda = True if torch.cuda.is_available() else False
cfg = Config()
data = Data()
data.get_images_list()
data.get_annotations()

# Very Importent!!!
# set config.py 'batch_size' to [ 1 ]
# then run this function
def predict(image_name, model_path=None):
    print(len(data.dictionary))
    encoder = Encoder()
    decoder = DecoderWithAttention(len(data.dictionary))
    if cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    if model_path:
        print('Loading the parameters of model.')
        if cuda:
            encoder.load_state_dict(torch.load(model_path[0]))
            decoder.load_state_dict(torch.load(model_path[1]))
        else:
            encoder.load_state_dict(torch.load(model_path[0], map_location='cpu'))
            decoder.load_state_dict(torch.load(model_path[1], map_location='cpu'))
    encoder.eval()
    decoder.eval()

    image = cv2.imread(image_name)
    image = cv2.resize(image, (224,224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:,:,0] = gray
    image[:,:,1] = gray
    image[:,:,2] = gray
    image = image.astype(np.float32) / 255.0
    image = image.transpose([2,0,1]) 
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    if cuda:
        image = image.cuda()

    output = encoder(image)
    # print('encoder output:', output.size())
    sentences, alphas = beam_search(decoder, output)
    # print(sentences)
    show(image_name, sentences[0], alphas[0])

    for sentence in sentences:
        prediction = []
        for word in sentence:
            prediction.append(data.dictionary[word])
            if word == 2:
                break
        # print(prediction)
        prediction = ' '.join([word for word in prediction])
        print('The prediction sentence:', prediction)

def beam_search(decoder, encoder_output, parameter_B=3):
    print('Beam Searching.')
    sen_len = data.sentence_length
    dict_len = len(data.dictionary)

    labels = torch.zeros(parameter_B, sen_len).type(torch.LongTensor)
    new_labels = torch.zeros(parameter_B, sen_len).type(torch.LongTensor)
    alphas = torch.zeros(parameter_B, sen_len, 7*7)
    temp_alphas = torch.zeros(parameter_B, sen_len, 7*7)

    labels = labels.cuda() if cuda else labels
    new_labels = new_labels.cuda() if cuda else new_labels

    for word_index in range(sen_len):
        for label_index in range(parameter_B):
            label = torch.unsqueeze(labels[label_index], 0)
            predictions, alpha = decoder(encoder_output, label)
            predictions = predictions.squeeze(0)[word_index]
            temp_alphas[label_index, word_index] = alpha[0, word_index]
            # print(predictions.size())
            p_label = predictions if label_index == 0 else torch.cat([p_label, predictions], 0)
            if word_index == 0:
                break
        for label_index in range(parameter_B):
            # print(p_label.size())
            max_index = torch.max(p_label, 0)[1]
            print(max_index)
            # print(p_label[0, max_index])
            # print(max_index)
            max_position = [int(max_index / dict_len), max_index % dict_len]
            # print(max_position)
            new_labels[label_index] = labels[max_position[0]]
            new_labels[label_index, word_index] = max_position[1]
            alphas[label_index] = temp_alphas[label_index]
            p_label[max_index] = -9999
            # print(p_label[0, max_index])
        labels = new_labels
        temp_alphas = alphas
        print(labels)
        # print(labels[:,word_index].cpu().data.numpy())
        if not labels[:,word_index].cpu().data.numpy().any():
            break

    return labels, alphas

def show(image_name, sentence, alphas):
    dictionary = data.dictionary
    image = cv2.imread(image_name)
    image = image[:,:,::-1]
    plt.figure(1)
    plt.imshow(image, interpolation='nearest', origin='upper')
    plt.title('source image')
    plt.xticks(())
    plt.yticks(())
    subplot_list = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    plt.figure(2)
    for i, word_ind in enumerate(sentence):
        if word_ind == 2:
            break
        if i == 9:
            break
        word = dictionary[word_ind]
        alpha = alphas[i].cpu().data.numpy().reshape(7,7)
        plt.subplot(subplot_list[i])
        plt.imshow(alpha, interpolation='bicubic', cmap=plt.cm.hot, origin='upper')
        plt.title(word)
        plt.xticks(())
        plt.yticks(())
    plt.show()

    







if __name__ == '__main__':
    predict('test.jpg', ['./models/train/encoder_resnet_20000.pkl', './models/train/decoder_20000.pkl']) 
    # predict('./data/RSICD/test/00029.jpg', ['./models/train/encoder_resnet_50000.pkl', './models/train/decoder_50000.pkl'])
