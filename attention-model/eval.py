import numpy as np 
import torch 
from dataloader import DataLoader 

cuda = True if torch.cuda.is_available() else False

def beam_search(data, decoder, encoder_output, parameter_B=3):
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
            predictions, alpha = decoder(encoder_output, label, is_train=False)
            predictions = predictions.squeeze(0)[word_index]
            temp_alphas[label_index, word_index] = alpha[0, word_index]
            # print(predictions.size())
            p_label = predictions if label_index == 0 else torch.cat([p_label, predictions], 0)
            if word_index == 0:
                break
        for label_index in range(parameter_B):
            # print(p_label.size())
            max_index = torch.max(p_label, 0)[1]
            # print(max_index)
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
        # print(labels)
        # print(labels[:,word_index].cpu().data.numpy())
        if not labels[:,word_index].cpu().data.numpy().any():
            break

    return labels, alphas

def cal_bleu(reference, candidate):
    """
    Calculate BLEU-4 score
    :param reference: ground-truth, a list, 5 sentences(5 str)
    :param candidate: prediction, a sentence, str
    """
    pr = cal_pr(candidate, reference)
    bp = cal_bp(candidate, reference)
    return pr * bp 

def get_grams(candidate, n):
    """
    Get all of n-gram
    :param candidate: prediction, a sentence, str
    :param n: 'n' of n-gram
    """
    words = candidate.split(' ')
    # print(words)
    grams = list()
    for i in range(len(words) - n + 1):
        # print(words[i:i+n])
        grams.append(' '.join(words[i:i+n]))
    return grams

def count_clip(gram, grams, reference):
    """
    Count clip of a gram
    Reference: https://blog.csdn.net/qq_31584157/article/details/77709454
    :param gram: a n-gram, str
    :param grams: all n-grams(n fixed), a list(str)
    :param reference: ground-truth, a list, 5 sentences(5 str)
    """
    clip = 0
    n = len(gram.split(' '))
    count_wi = 0
    for g in grams:
        if gram == g:
            count_wi += 1
    # print('count_wi:', count_wi)
    for ref in reference:
        ref_list = ref.split(' ')
        count_ref = 0
        for i in range(len(ref_list) - n + 1):
            if gram == ' '.join(ref_list[i:i+n]):
                count_ref += 1
        # print('count_ref: ', count_ref)
        count = min(count_wi, count_ref)
        if count > clip:
            clip = count
    return clip

def cal_pn(grams_set, grams, candidate, reference):
    """
    Calculate pn(p1, p2, p3, p4)
    :param grams_set: a set of grams, set(str)
    :param grams: all n-grams(n fixed), a list(str)
    :param candidate: prediction, a sentence, str
    :param reference: ground-truth, a list, 5 sentences(5 str)
    """
    count = 0
    for gram in grams_set:
        # print(gram)
        count += count_clip(gram, grams, reference)
    # calculate log() for p, so '+10**-8' avoid 'p==0'
    p = count / len(grams) + 10**-8 
    return p

def cal_pr(candidate, reference):
    """
    Calculate precision.
    :param candidate: prediction, a sentence, str
    :param reference: ground-truth, a list, 5 sentences(5 str)
    """
    pn = list()
    for n in range(1, 2): # modify this line to set calculate BLEU-N
        grams = get_grams(candidate, n)
        if grams == []:
            break
        # print(grams)
        grams_set = set(grams)
        pn.append(cal_pn(grams_set, grams, candidate, reference))
    # print(pn)
    pr = np.exp(np.log(pn).mean())
    # print(pr)
    return pr

def cal_bp(candidate, reference):
    """
    Calculate brevity penalty.
    Reference: https://www.cnblogs.com/by-dream/p/7679284.html
    :param candidate: prediction, a sentence, str
    :param reference: ground-truth, a list, 5 sentences(5 str)
    """
    dis_min = 100
    len_ref = 0
    for ref in reference:
        if abs(len(ref) - len(candidate)) < dis_min:
            dis_min = abs(len(ref) - len(candidate))
            len_ref = len(ref)
    if len_ref < len(candidate):
        bp = 1
    else:
        bp = np.exp(1 - len_ref / len(candidate))
    # print(bp)
    return bp
        
def val_eval(encoder, decoder, dataloader):
    print('Eavluating..')
    images = dataloader.data.images
    annotations = dataloader.data.annotations
    val_list = dataloader.data.val_list
    np.random.shuffle(val_list)
    index = val_list[:100]
    encoder.eval()
    decoder.eval()
    bleu = list()
    for ind in index:
        image = images[ind]
        image = image.astype(np.float32) / 255.0
        image = image.transpose([2,0,1]) 
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        reference = annotations[ind]
        # print(image.shape)
        # print(reference)
        output = encoder(image)
        sentences, _ = beam_search(dataloader.data, decoder, output)
        prediction = []
        for word in sentences[0]:
            prediction.append(dataloader.data.dictionary[word])
            if word == 2:
                break
        prediction = ' '.join([word for word in prediction])
        # print(prediction)
        # print(reference)
        bleu4 = cal_bleu(reference, prediction)
        bleu.append(bleu4)
    encoder.train()
    decoder.train()
    print('BLEU4: ', bleu)
    return np.array(bleu).mean()

def test_eval(encoder, decoder, data):
    encoder.eval()
    decoder.eval()
    images = data.images
    annotations = data.annotations
    test_list = data.test_list
    bleu = list()
    for ind in test_list:
        image = images[ind]
        image = image.astype(np.float32) / 255.0
        image = image.transpose([2,0,1]) 
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        reference = annotations[ind]
        # print(image.shape)
        # print(reference)
        output = encoder(image)
        sentences, _ = beam_search(data, decoder, output)
        prediction = []
        for word in sentences[0]:
            prediction.append(data.dictionary[word])
            if word == 2:
                break
        prediction = ' '.join([word for word in prediction])
        bleu4 = cal_bleu(reference, prediction)
        bleu.append(bleu4)
    np.save('./result/test_bleu1.npy', bleu)

if __name__ == '__main__':
    bleu4 = cal_bleu(['today is a nice day a a a a'], 'it is a nice day today')
    print(bleu4)


    