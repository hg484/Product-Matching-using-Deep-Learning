import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL,cv2
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import EfficientNetB4,ResNet101,EfficientNetB5
from sklearn.model_selection import train_test_split

SEED=48
DEBUG=False

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH='../input/shopee-product-matching/'
train=pd.read_csv(BASE_PATH+"train.csv")
train['image_path']=BASE_PATH+'train_images/'+train.image

id_to_label_mapping=dict(zip(train.label_group.unique(),range(train.label_group.nunique())))
train["label_number"]=train.label_group.map(id_to_label_mapping)

NUM_CLASSES=train.label_group.nunique()
HEIGHT,WIDTH=256,256
CHANNELS=3
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=32

print("Sample of Available Data")
print(train.head(),'\n')

# Data processing function for creating tf.data dataset 
# Converting image path dataset to image label dataset
def process_data(image_path,label):
    img=tf.io.read_file(image_path)
    img=tf.image.decode_jpeg(img,channels=CHANNELS)
    img=tf.image.resize(img,[HEIGHT,WIDTH])
    return img,label

# function to improve dataset processing speed 
def configure_for_performance(ds,batch_size):
    ds=ds.cache('/kaggle/dump.tfcache')
    
    ds=ds.shuffle(buffer_size=1024)
    ds=ds.batch(BATCH_SIZE)
    ds=ds.prefetch(buffer_size=AUTOTUNE)
    return ds

x_train,x_valid=train_test_split(train,test_size=0.1,random_state=SEED,shuffle=True)

# image path & label dataset
train_ds=tf.data.Dataset.from_tensor_slices((x_train.image_path.values,x_train.label_number.values))
valid_ds=tf.data.Dataset.from_tensor_slices((x_valid.image_path.values,x_valid.label_number.values))

# image & label dataset
train_ds=train_ds.map(process_data,num_parallel_calls=AUTOTUNE)
valid_ds=valid_ds.map(process_data,num_parallel_calls=AUTOTUNE)

# improving dataset by shuffling dataset,creating image batch and prefetching dataset
# more information : https://www.tensorflow.org/guide/data_performance
# Batch image and label dataset
train_ds_batch = configure_for_performance(train_ds, 8)
valid_ds_batch = valid_ds.batch(8)

print("Dataset before reconfiguring")
print("train data:",train_ds.cardinality())
print("valid data",valid_ds.cardinality(),'\n')

print("Dataset after reconfiguring(BATCH+SHUFFLE+PREFETCH)")
print("train data:",train_ds_batch.cardinality())
print("valid data",valid_ds_batch.cardinality(),'\n')

# Function to get our f1 score
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

# loss function 
# https://github.com/AdrianUng/keras-triplet-loss-mnist#:~:text=Triplet%20Loss%20explained%3A,-Figures%20taken%20from&text=By%20pairing%20the%20images%20into,respect%20to%20all%20other%20classes.&text=Where%20d(A%2CP),the%20Positive%20and%20Negative%20pairs.

def pairwise_distances(embeddings):
    dot_product = tf.linalg.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.math.maximum(distances, 0.0)

    mask = tf.cast(tf.equal(distances, 0.0),tf.float32)
    distances = distances + mask * 1e-16
    distances = tf.math.sqrt(distances)
    distances = distances * (1.0 - mask)

    return distances

def get_anchor_positive_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)

    labels_equal = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.math.logical_and(indices_not_equal, labels_equal)

    return mask

def get_anchor_negative_triplet_mask(labels):
    labels_equal = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.math.logical_not(labels_equal)

    return mask

def get_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.math.logical_and(tf.math.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    label_equal = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.math.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.math.logical_and(distinct_indices, valid_labels)

    return mask


class TripletLossFn(tf.keras.losses.Loss):
    def __init__(self,margin=1.0,**kwargs):
        super().__init__(**kwargs)
        self.margin = margin
  
    def call(self,y_true,y_pred):

        labels = tf.convert_to_tensor(y_true)
        labels = tf.squeeze(labels,axis=-1)
        embeddings = tf.convert_to_tensor(y_pred)

        pairwise_dist = pairwise_distances(embeddings)

        mask_anchor_positive = get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.cast(mask_anchor_positive,tf.float32)

        anchor_positive_dist = tf.math.multiply(mask_anchor_positive, pairwise_dist)

        hardest_positive_dist = tf.math.reduce_max(anchor_positive_dist, axis=1, keepdims=True)


        mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.cast(mask_anchor_negative,tf.float32)

        max_anchor_negative_dist = tf.math.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)


        hardest_negative_dist = tf.math.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    

        triplet_loss = tf.math.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)

        triplet_loss = tf.math.reduce_mean(triplet_loss)

        return triplet_loss
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,"margin":self.margin}

# Model creation using a pretrained model
def create_model(pretrained_model):  
    
    model=tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    
    return model

# funtion to compile a choosen model
def compile_model(model,LR=0.0001):
    
    optimizer = tf.keras.optimizers.Adam(lr=LR)
    
    loss=TripletLossFn(0.7)
    
    metrics = [
       tf.keras.metrics.SparseCategoricalAccuracy()
    ]

    model.compile(optimizer=optimizer, loss=loss)
    
    return model

# Callback list generation
def callback_creation(model_path):
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.1,
        patience=3,
        verbose=0
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10, 
        verbose=1
    )
    
    callbacks=[reduce_lr,model_checkpoint,early_stopping]
    
    return callbacks

# model training
def model_training(pretrained_model,model_path):
    EPOCH_COUNTS=50
    VERBOSE=1
    LR=0.0001

    tf.keras.backend.clear_session();

    model=create_model(pretrained_model)
    model=compile_model(model,LR=LR)
    callback_list=callback_creation(model_path)


    history=model.fit(
                        train_ds_batch,
                        validation_data=valid_ds_batch,
                        epochs=EPOCH_COUNTS,
                        callbacks=callback_list,
                    )

# Same script for model training different scipts due to problem with GPU configuration
# individual model training

efnb4_model_path='./best_model_efnb4.h5'
pretrained_model=EfficientNetB4(include_top=False, weights='imagenet',pooling='avg',input_shape=[HEIGHT,WIDTH, 3])
model_training(pretrained_model,efnb4_model_path)
