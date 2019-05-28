import numpy as np 
import torch 
from torch.autograd import Variable
from config import Config
from model import Encoder, DecoderWithAttention
from dataloader import DataLoader 
from augmentations import Augmentation
from eval import val_eval

cuda = True if torch.cuda.is_available() else False
cfg = Config()

def cal_loss(sentences, batch_label, alphas, alpha_c):
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(sentences.size(1)):
        label = batch_label[:, i]
        word = sentences[:,i,:]
        # print(label.size()[0])
        if i == 0:
            loss = loss_func(word, label)
        else:
            loss += loss_func(word, label)
    loss = loss / (i+1)
    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    return loss

def train(model_path=None):
    dataloader = DataLoader(Augmentation())
    encoder = Encoder()
    dict_len = len(dataloader.data.dictionary)
    decoder = DecoderWithAttention(dict_len)

    if cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    # if model_path:
    #   text_generator.load_state_dict(torch.load(model_path))
    train_iter = 1
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.encoder_learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_learning_rate)

    val_bleu = list()
    losses = list()
    while True:
        batch_image, batch_label = dataloader.get_next_batch()
        batch_image = torch.from_numpy(batch_image).type(torch.FloatTensor)
        batch_label = torch.from_numpy(batch_label).type(torch.LongTensor)
        if cuda:
            batch_image = batch_image.cuda()
            batch_label = batch_label.cuda()
        # print(batch_image.size())
        # print(batch_label.size())

        print('Training')
        output = encoder(batch_image)
        # print('encoder output:', output.size())
        predictions, alphas = decoder(output, batch_label)

        loss = cal_loss(predictions, batch_label, alphas, 1)

        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

        print(
            'Iter', train_iter, 
            '| loss:', loss.cpu().data.numpy(), 
            '| batch size:', cfg.batch_size,
            '| encoder learning rate:', cfg.encoder_learning_rate, 
            '| decoder learning rate:', cfg.decoder_learning_rate
        )
        losses.append(loss.cpu().data.numpy())
        if train_iter % cfg.save_model_iter == 0:
            val_bleu.append(val_eval(encoder, decoder, dataloader))
            torch.save(encoder.state_dict(), './models/train/encoder_'+cfg.pre_train_model+'_'+str(train_iter)+'.pkl')
            torch.save(decoder.state_dict(), './models/train/decoder_'+str(train_iter)+'.pkl')
            np.save('./result/train_bleu4.npy', val_bleu)
            np.save('./result/losses.npy', losses)

        if train_iter == cfg.train_iter:
            break
        train_iter += 1
    




if __name__ == '__main__':
    train()
