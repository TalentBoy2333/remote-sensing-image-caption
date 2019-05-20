class Config():
    images_folder = './data/RSICD/RSICD_images/'
    annotations_name = './data/RSICD/dataset_rsicd.json'

    # pretrain model config
    pre_train_model = 'mobilenet'
    fix_pretrain_model = False
    feature_size = 1280 # pretrain model's feature map number in final layer

    # Attention layer config
    attention_size = 1280

    # LSTM config 
    embed_size = 1280
    input_size = embed_size + feature_size # encoder output feature vector size: 1280
    hidden_size = 1280 # 4096
    num_layers = 1

    # training config
    batch_size = 2 # 64
    train_iter = 30001 # 100000
    encoder_learning_rate = 1e-4
    decoder_learning_rate = 1e-4
    save_model_iter = 5000