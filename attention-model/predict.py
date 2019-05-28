import numpy as np 
import torch 
from config import Config
from model import Encoder, DecoderWithAttention
from data import Data 
from eval import beam_search, test_eval
import cv2
import matplotlib.pyplot as plt 

cuda = True if torch.cuda.is_available() else False
cfg = Config()
data = Data()

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
    image = image.astype(np.float32) / 255.0
    image = image.transpose([2,0,1]) 
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    if cuda:
        image = image.cuda()

    output = encoder(image)
    # print('encoder output:', output.size())
    sentences, alphas = beam_search(data, decoder, output)
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
    # predict('./data/RSICD/RSICD_images/00110.jpg', ['./models/train/encoder_mobilenet_60000.pkl', './models/train/decoder_60000.pkl']) 
    # predict('./data/RSICD/test/00029.jpg', ['./models/train/encoder_resnet_50000.pkl', './models/train/decoder_50000.pkl'])

    model_path = ['./models/train/encoder_mobilenet_60000.pkl', './models/train/decoder_60000.pkl']
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
    test_eval(encoder, decoder, data)
