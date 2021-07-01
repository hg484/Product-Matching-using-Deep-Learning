import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,re
import PIL,cv2,math
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import EfficientNetB4,ResNet101,EfficientNetB5
from sklearn.model_selection import train_test_split


try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

SEED=48
DEBUG=False

# Seeding
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH='../input/shopee-product-matching/'
train=pd.read_csv(BASE_PATH+"train.csv")
train['image_path']=BASE_PATH+'train_images/'+train.image

id_to_label_mapping=dict(zip(train.label_group.unique(),range(train.label_group.nunique())))
train["label_number"]=train.label_group.map(id_to_label_mapping)

GCS_PATH = KaggleDatasets().get_gcs_path('shopee-tf-records-512-stratified')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*.tfrec')

NUM_CLASSES=11014 # train.label_group.nunique()
HEIGHT,WIDTH=256,256
CHANNELS=3
AUTO=AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

# print("Sample of Available Data")
# print(train.head(),'\n')

# Data processing function for creating tf.data dataset 
# Converting image path dataset to image label dataset
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, [HEIGHT,WIDTH])
    #image = tf.cast(image, tf.float32) / 255.0
    return image

# Converting the tf.data.dataset in a manner  
# which  usable with arcface layer
def arcface_format(image,label):
    return {'image_input':image,'label_input':label},label

# Getting specific data from the tfrecord file
# function returns image and label
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "posting_id": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label_group": tf.io.FixedLenFeature([], tf.int64),
        "matches": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label_group = tf.cast(example['label_group'], tf.int32)
    return image,label_group

# Possible data augmentations
def image_augment(image,label):
    
    image=tf.image.random_brightness(image, 0.1)
    image=tf.image.random_contrast(image, 0.8, 1.2)
    #image=tf.image.random_distort_color(image)
    
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    
    return image,label
    

# Difference between training and valid dataset preparation is shuffling of the training data
# Shuffling allows the model finetuning to be robust
def get_training_dataset(filenames, ordered = False):
    dataset = load_dataset(filenames, ordered = ordered)
    #dataset = dataset.map(image_augment,num_parallel_calls = AUTOTUNE)
    
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTOTUNE)
        
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    
    dataset = configure_for_performance(dataset)
    return dataset

def get_validation_dataset(filenames, ordered = True):
    dataset = load_dataset(filenames, ordered = ordered)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTOTUNE)
    dataset = configure_for_performance(dataset)
    return dataset

# Improving the efficiency of model training
def configure_for_performance(dataset):
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(AUTOTUNE)
    return dataset

# Open the tfrecord data and return image_label dataset 
def load_dataset(filenames, ordered = False):
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = AUTOTUNE) 
    return dataset

def count_data_items(filenames):
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


'''
Implements large margin arc distance.

Reference:
    https://arxiv.org/pdf/1801.07698.pdf
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
        blob/master/src/modeling/metric_learning.py
'''

class ArcMarginProduct(tf.keras.layers.Layer):

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# Model Creation function
def create_model(pretrained_model):  
    
    with strategy.scope():
        
        margin = ArcMarginProduct(
                n_classes = NUM_CLASSES, 
                s = 30, 
                m = 0.5, 
                name='head/arc_margin', 
                dtype='float32'
                )

        inp = tf.keras.layers.Input(shape = (HEIGHT,WIDTH, 3), name = 'image_input')
        label = tf.keras.layers.Input(shape = (), name = 'label_input')
        x = pretrained_model(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = margin([x, label])

        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])

        optimizer = tf.keras.optimizers.Adam(lr=0.00001)
        loss=tf.keras.losses.SparseCategoricalCrossentropy()


        metrics = [
           tf.keras.metrics.SparseCategoricalAccuracy()
        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    
#Callback list generation
# First callback : decreasing learning rate
# def get_lr_callback():
#     def scheduler(epoch,lr):
#         if(lr%4==0):
#             lr*tf.math.exp(-1)
#             return lr
#         else:
#             return lr
        
#     lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = True)
#     return lr_callback

def get_lr_callback():
    lr_start   = 0.000001
    lr_max     = 0.000005 * BATCH_SIZE
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max    
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min    
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    return lr_callback

# Function generating callback list for the model
def callback_creation(model_path):
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='sparse_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='sparse_categorical_accuracy',
        min_delta=0.1,
        mode='max',
        patience=4, 
        verbose=1
    )
    
    callbacks=[get_lr_callback(),model_checkpoint]
    
    return callbacks

# model training
def model_training(pretrained_model,model_path,train,valid,TRAIN_COUNT):
    # EPOCH_COUNTS=50
    EPOCH_COUNTS=20
    VERBOSE=1
    LR=0.00001

    tf.keras.backend.clear_session();

    model=create_model(pretrained_model)
    callback_list=callback_creation(model_path)

    history=model.fit(
                        train,
                        epochs=EPOCH_COUNTS,
                        callbacks=callback_list,
                        validation_data=valid,
                        verbose=VERBOSE,
                        steps_per_epoch = TRAIN_COUNT // BATCH_SIZE
                    )


# Data spliting
train_split,valid_split=train_test_split(TRAINING_FILENAMES, shuffle = True, random_state = SEED,test_size=0.25)

# training and validation data generation
train_dataset = get_training_dataset(train_split, ordered = False)
val_dataset = get_validation_dataset(valid_split, ordered = True)    

# count of training and valid data 
TRAIN_COUNT=count_data_items(train_split)
VALID_COUNT=count_data_items(valid_split)


with strategy.scope():
    efnb5_model_path='./best_model_efnb5.h5'
    pretrained_model=EfficientNetB5(include_top=False, weights='imagenet')
    
model_training(pretrained_model,efnb5_model_path,train_dataset,val_dataset,TRAIN_COUNT)

with strategy.scope():
    efnb4_model_path='./best_model_efnb4.h5'
    pretrained_model=EfficientNetB4(include_top=False, weights='imagenet')
    
model_training(pretrained_model,efnb4_model_path,train_dataset,val_dataset,TRAIN_COUNT)

with strategy.scope():
    resnet101_model_path='./best_model_resnet101.h5'
    pretrained_model=ResNet101(include_top=False, weights='imagenet')
    
model_training(pretrained_model,resnet101_model_path,train_dataset,val_dataset,TRAIN_COUNT)
