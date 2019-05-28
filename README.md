This is a project of remote sensing image.</br>

* Classification of remote sensing image
* Remote sensing image caption

I'm using `torch 0.4`, `opencv-python`, `numpy`, `matplotlib` in `python 3.6`</br>

## Classification
### Data
I used `NWPU-RESISC45` dataset, you can download this dataset(http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) to train your model, or you can download other dataset, but you should pay attention to the difference of the way to load data from daataset, we maybe used different way to load data.</br>
All in all, check the `dataset.py`.</br>

### Model
We can choose two pre-train models, `resnet_v101` and `mobilenet_v2`.
```Python
classifier = Classifier(model_name, class_number, True)
```
Then, I just add a full connect layer and softmax to classify.</br>

### How to use
If you want to train you own model, you just need to prepare the dataset, then run the `train.py`.
```Bash
python train.py
```
By the way, you can modify the `train.py`, to set `pre-train model`, `batch size`, `epoch`, `learning rate`, and continue training base on the `model which was saved in last training`.</br>
The training model will be saved in `./models/train/`.</br></br>
If you want to predict the class of a new remote sensing image.</br>
First, you should modify the `predict.py` to set image name, load the parameters of model and you can also see the display of result.
```Python
image_name = 'test.jpg'
classifier = get_classifier('mobilenet', './models/train/classifier_50.pkl')
predict(classifier, image_name, True)
```
Then, just run the `predict.py`.
```Bash
python predict.py
```

## Image Caption
### Data
I used the `RSICD` dataset</br>
`Lu X, Wang B, Zheng X, et al. Exploring Models and Data for Remote Sensing Image Caption Generation[J]. IEEE Transactions on Geoscience and Remote Sensing, 2017.` </br>
to train my model, you can download this dataset at https://github.com/201528014227051/RSICD_optimal</br>
Or, you can use your own dataset, same as `Classification`, modify the `data.py` and `dataloader.py` for your dataset.</br>
### Model
I used `Show, Attend and Tell` model, you can read this paper: `Xu, Kelvin, et al. “Show, attend and tell: Neural image caption generation with visual attention.” arXiv preprint arXiv:1502.03044 (2015).`, or you can refer to https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning</br>
My model incloud `encoder`, `decoder` and `attention`.</br>
Because of our project is used on ARM, so we must simplify the network, our encoder is `mobilenet_v2`, we delete the full connect layer for classification, make the `mobilenet_v2` output the feature map of image(size 7*7).</br></br>
Attention part is composed of some full connect layer, input is the hidden layer's output of decoder, output is a tensor(size 1*49), reshape this tensor to size 7*7.</br>
Then, we can get feature vector by `attention tenser(7*7)` and `feature map(7*7)`, and this feature vector is the input of decoder.</br>
Attention image: 
![](https://github.com/TalentBoy2333/remote-sensing-image-caption/blob/master/images/attention.png)</br>
Decoder is base on LSTM, input is the embedding of words in dictorary(every word in dictionary is a `one-hot` code, and they will be transformed to feature vector by embedding layer), hidden layer is connected with full connect layer and softmax, output is the probability of the next word.</br>
By the way, first input is a signal of beginning: `<start>`, and the last output is a signal of endding: `.`.</br>
### Data Augmentation
I used a similar approach to `SSD`(`https://arxiv.org/abs/1512.02325`), in evary iteration, I change the value of random pixels of the mini-batch, add random lighting noise, randomly swap image channels, randomly adjust the contrast of image. Then, I randomly crop a part of the image sample in mini-batch and randomly mirror the image after sample cropping.
### Train
If you want to train your model, make sure that you have the `RSICD` dataset, if your dataset is different from `RSICD`, you should modify `data.py` and `dataloader.py` for your data.</br>
if you want to change the details of the model, you should modify `model.py` and `config.py`.</br>
In `config.py`, you can also modify the `learning rate`, `batch size`, `epoch` and so on.</br>
Then, we can start training by running `train.py`, you can modify the function `train()` to decide to training from nothing or traning from last model parameters.
```Bash
python train.py
```
Loss function curve: 
![](https://github.com/TalentBoy2333/remote-sensing-image-caption/blob/master/images/loss.png)
### Predict
I used beam search to find the best sentence of image caption because beam search consider more possibility.</br>
> Beam Search(Assuming that the dictionary is [a, b, c], beam size chooses 2):
>> Step 1: When generating the first word, choose the two words with the highest probability, then the current sequence is `a` or `b`.</br></br>
>> Step 2: When the second word is generated, we combine the current sequence `a` or `b` with all the words in the dictionary to get six new sequences `aa`, `ab`, `ac`, `ba`, `bb`, `bc`, and then select two of them with the highest probability as the current sequence, `ab` or `bb`.</br></br>
>> Step 3: Repeat this process until the terminator(`'.'`) is encountered. The final output is two sequences with the highest probability.</br>

I set the parameter of beam search to `3`, you can modify the parameter in `eval.py`.
```Python
def beam_search(data, decoder, encoder_output, parameter_B=3):
```
If you want to predict an image, you should modify `predict.py` to set test image name and the path of model parameters.
```Python
predict('test.jpg', ['./models/train/encoder_mobilenet_20000.pkl', './models/train/decoder_20000.pkl'])
```
Then, you can run `predict.py`, and you will see the `image`, `sentence of image caption` and `the distribution image` of attention module for every word.
```Bash
python predict.py
```
Example: 
image: 
![](https://github.com/TalentBoy2333/remote-sensing-image-caption/blob/master/images/image.png)
caption:
![](https://github.com/TalentBoy2333/remote-sensing-image-caption/blob/master/images/caption.png)
### Evaluation
I use `BLEU-4` to evaluate the quality of generated sentences.</br>
BLEU-4: http://xueshu.baidu.com/s?wd=paperuri%3A%2888a98dec5bea94cca9f474db30c36319%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fciteseer.ist.psu.edu%2Fviewdoc%2Fdownload%3Bjsessionid%3DF4B7103527B9E68CE036BB1F77EB78BD%3Fdoi%3D10.1.1.19.9416%26rep%3Drep1%26type%3Dpdf&ie=utf-8&sc_us=137105618768529979</br></br>
My training BLEU: </br>
BLEU-1: 0.4899</br>
BLEU-2: 0.2312</br>
BLEU-3: 0.1003</br>
BLEU-4: 0.0432</br>