from sacred import Experiment

ex =  Experiment("baseline")


@ex.config
def config():
    exp_name = 'CrossBaseline'
    seed = 0
    dataset_name = 'flickr25'  # flickr coco nuswide and iaprtc
    batch_size = 32
    test_batch_size = 64

    crop_size = 224
    max_epochs = 300
    hash_length = 32

    #optimizer config
    lr =  1e-5
    image_lr = 1e-4
    text_lr = 1e-5
    momentum = 0.9
    weight_decay = 0.0005
    margin = 15


    #text config
    text_dim = 1386
    hidden_dim = 8192

    #image_Config
    pretrained_dir = './pretrained_dir/imagenet-vgg-f.mat'


    #evaluate interval
    eval_interval = 10

    #gpu training
    num_gpus = 1
    precision = 16

    #log config
    log_dir = './results/'
    log_interval = 10
    val_check_interval =  313

    #from checkpoint
    load_path = ''

    #directories
    data_root = 'data/mirflicker25k'

    test_only = False
