from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = True
config.output = 'edgeface_s_gamma_05/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 1024
config.optimizer = "adamw"
config.lr = 1e-5
config.verbose = 2000
config.dali = True 
config.dali_aug = True

config.num_workers = 6

config.rec = "data/webface12m"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = []

