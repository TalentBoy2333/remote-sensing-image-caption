import numpy as np 
import torch 
from torch.autograd import Variable
from config import Config
from MobileNetV2 import MobileNetV2
from dataloader import DataLoader

cfg = Config()
cuda = True if torch.cuda.is_available() else False

class Encoder(torch.nn.Module):
    """
    Encoder.
    """
    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # resnet = resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(resnet.children())[:-2]
        # self.resnet = torch.nn.Sequential(*modules)

        mobilenet = MobileNetV2(n_class=1000)
        state_dict = torch.load('./models/mobilenet_v2.pth.tar', map_location='cpu') # add map_location='cpu' if no gpu
        mobilenet.load_state_dict(state_dict)
        self.mobilenet = mobilenet.features
           
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.mobilenet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(torch.nn.Module):
    """
    Attention Network.
    """

    def __init__(self):
        super(Attention, self).__init__()
        self.encoder_att = torch.nn.Linear(cfg.feature_size, cfg.attention_size)  # linear layer to transform encoded image
        self.decoder_att = torch.nn.Linear(cfg.hidden_size, cfg.attention_size)  # linear layer to transform decoder's output
        self.full_att = torch.nn.Linear(cfg.attention_size, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(torch.nn.Module):
    """
    Decoder.
    """

    def __init__(self, dict_length):
        """
        :param dict_length: size of data's dictionary.
        """
        super(DecoderWithAttention, self).__init__()
        self.dict_length = dict_length
        self.attention = Attention()  # attention network

        # nn.Embedding: change a int number in [0, dict_length] to a vector size(cfg.embed_size).
        self.embedding = torch.nn.Embedding(dict_length, cfg.embed_size)  # embedding layer
        self.dropout = torch.nn.Dropout(p=0.5)
        self.decode_step = torch.nn.LSTMCell(cfg.input_size, cfg.hidden_size, bias=True)  # decoding LSTMCell
        # encoder's output feature vector size(2048), change it to hidden_size.
        self.init_h = torch.nn.Linear(cfg.feature_size, cfg.hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = torch.nn.Linear(cfg.feature_size, cfg.hidden_size)  # linear layer to find initial cell state of LSTMCell
        # create a gate vector to choose the more importent cell of the feature vector.
        self.f_beta = torch.nn.Linear(cfg.hidden_size, cfg.feature_size)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(cfg.hidden_size, dict_length)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_tune_embeddings()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, is_train=True):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # Flatten image
        if is_train:
            encoder_out = encoder_out.view(cfg.batch_size, -1, cfg.feature_size)  # (batch_size, num_pixels, encoder_dim)
        else:
            encoder_out = encoder_out.view(1, -1, cfg.feature_size)
        num_pixels = encoder_out.size(1)

        # Embedding
        sentence_length = encoded_captions.size(1)

        """
        # torch.ones(): create a number of [1].
        # label:
            [   9.  691.  241.   18.   41.    9.   66.   27.  262.   22.    9.   11.
            27.   32.   34.   35.    2.]
        # We want input is:
            [   1.    9.  691.  241.   18.   41.    9.   66.   27.  262.   22.    9. 
            11.   27.   32.   34.   35.]
        # So, we take label[:-1]:
            [   9.  691.  241.   18.   41.    9.   66.   27.  262.   22.    9.   11.
            27.   32.   34.   35.]
        # Then, concatenate torch.ones() and label[:-1].
        """
        if is_train:
            prewords_start = torch.ones(cfg.batch_size, 1).type(torch.LongTensor)
        else:
            prewords_start = torch.ones(1, 1).type(torch.LongTensor)
        prewords_start = prewords_start.cuda() if cuda else prewords_start
        prewords_behind = encoded_captions[:,:-1]
        prewords_label = torch.cat([prewords_start, prewords_behind], 1)
        embeddings = self.embedding(prewords_label)  # (batch_size, sentence_length, embed_dim)
        # print('embeddings:', embeddings.size())
        # print('sentence length:', sentence_length)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # print('h:', c.size())
        # print('c:', h.size())

        # Create tensors to hold word predicion scores and alphas
        if is_train:
            predictions = torch.zeros(cfg.batch_size, sentence_length, self.dict_length)
            alphas = torch.zeros(cfg.batch_size, sentence_length, num_pixels)
        else:
            predictions = torch.zeros(1, sentence_length, self.dict_length)
            alphas = torch.zeros(1, sentence_length, num_pixels)
        if cuda:
            predictions = predictions.cuda()
            alphas = alphas.cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for i in range(sentence_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            # print('attention output:', attention_weighted_encoding.size())
            # print('alpha:', alpha.size())
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([embeddings[:, i, :], attention_weighted_encoding], 1), (h, c)) # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            # print('predictions:', preds.size())
            predictions[:, i, :] = preds
            alphas[:, i, :] = alpha

        return predictions, alphas




if __name__ == '__main__':
    dataloader = DataLoader()
    batch_image, batch_label = dataloader.get_next_batch()
    print(batch_image.shape)
    print(batch_label.shape)
    batch_image = torch.from_numpy(batch_image).type(torch.FloatTensor)
    batch_label = torch.from_numpy(batch_label).type(torch.LongTensor)

    encoder = Encoder()
    dict_len = len(dataloader.data.dictionary)
    rnn = DecoderWithAttention(dict_len)

    output = encoder(batch_image)
    print('encoder output:', output.size())

    predictions, alphas = rnn(output, batch_label)
    print('prediction:', predictions.size())
    print('alphas:', alphas.size())

