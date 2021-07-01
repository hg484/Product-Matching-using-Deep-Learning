# ! pip install cudf
# ! pip install cuml
# ! pip install cupy

import numpy as np
import pandas as pd
import gc,cv2,math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4,ResNet101,EfficientNetB5
from sklearn.metrics.pairwise import cosine_similarity

# For this project i am using the RAPIDS Library developed by NVIDIA
# The RAPIDS suite of open source software libraries and APIs gives you the ability 
# to execute end-to-end data science and analytics pipelines entirely on GPUs
# more information: https://rapids.ai/
import cudf,cuml,cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

print("Library versions ->")
print("RAPIDS:",cuml.__version__)
print("TF:",tf.__version__,'\n')

# Limit tensorflow memory usage to 1GB leaving rest for RAPIDS
# RAPIDS Library requires GPU for extrememly fast computations when compared to numpy 
LIMIT=1
gpus=tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
        logical_gpus=tf.config.experimental.list_logical_devices("GPU")
        
        print(len(gpus),"Physical GPUs")
        print(len(logical_gpus),"Logical GPUs")
        
    except RuntimeError as e:
        print(e)
            
print(f"Tensorflow restricted to {LIMIT}GB RAM")
print(f"RAPIDS has {16-LIMIT}GB RAM available for use ")


HEIGHT,WIDTH=256,256
CHANNELS=3
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=32

# SETTING UP DATA
DATA_PATH='../input/shopee-product-matching/'
IMG_BASE=DATA_PATH+'train_images/'

test=pd.read_csv(DATA_PATH+"test.csv")
train=pd.read_csv(DATA_PATH+"train.csv")

trgt=train.groupby("label_group").posting_id.agg("unique")
train['target']=train.label_group.map(trgt)

train['image_path']=DATA_PATH+'train_images/'+train.image

id_to_label_mapping=dict(zip(train.label_group.unique(),range(train.label_group.nunique())))
train["label_number"]=train.label_group.map(id_to_label_mapping)

NUM_CLASSES=train.label_group.nunique()

# convert train pandas dataframe into cudf dataframe for RAPIDS use
train_gf = cudf.DataFrame(train)

# embeddings chunk size
CHUNK=1024*1
C_LEN=len(train)//CHUNK
if len(train)%CHUNK!=0: C_LEN += 1




# print("Sample of available data")
# print(f"Train Data shape is {train.shape}")
# print(train.head())

# metric for measuring accuracy of pipeline
# F1 score
def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

#  SETTING A BASELINE USING PHASH VALUES
IMG_PHASH=train.groupby('image_phash').posting_id.agg('unique').to_dict()
train['phash']=train.image_phash.map(IMG_PHASH)
train["f1"]=train.apply(getMetric("phash"),axis=1)
print(f"F1 Score for phash baseline = {train.f1.mean()}")


# Data Generator for faster and efficient usage of GPU for embeddings generations
# more information: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, img_size=256, batch_size=32, path=''): 
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path
        self.indexes = np.arange( len(self.df) )
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.df) // self.batch_size
        ct += int(( (len(self.df)) % self.batch_size)!=0)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indexes)
        return X
            
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        X = np.zeros((len(indexes),self.img_size,self.img_size,3),dtype='float32')
        df = self.df.iloc[indexes]
        for i,(index,row) in enumerate(df.iterrows()):
            img = cv2.imread(self.path+row.image)
            X[i,] = cv2.resize(img,(self.img_size,self.img_size)) #/128.0 - 1.0
        return X

# Converting image path dataset to a image dataset
def process_data(image_path,label):
    img=tf.io.read_file(image_path)
    img=tf.image.decode_jpeg(img,channels=CHANNELS)
    img=tf.image.resize(img,[HEIGHT,WIDTH])
    # img = tf.cast(img, tf.float32) / 255.0
    return img,label

# Converting tf.data.dataset in a manner to make it usable for arcface layer
def arcface_format(image,label):
    return {'image_input':image,'label_input':label},label

# Final dataset producing function using image path and label
def get_dataset(image,label):
    ds=tf.data.Dataset.from_tensor_slices((image,label))
    ds=ds.map(process_data,num_parallel_calls=AUTOTUNE)
    ds=ds.map(arcface_format,num_parallel_calls=AUTOTUNE)
    ds=ds.batch(8)
    
    return ds

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



# Model creation using a pretrained model
def create_model(pretrained_model):  
    
    model=tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    
    return model

# Arcface model creation function
def create_model_arcface(pretrained_model):  
    
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
    
    return model

# Function to generate embedding using sequential model function
def embeddings_generation_Normal(model):
    image_embeds=[]
    for i,j in enumerate(range(C_LEN)):
        a=j*CHUNK
        b=min((j+1)*CHUNK,len(train))
        print('CHUNK:',a,"to",b)

        curr_test_gen=DataGenerator(train.iloc[a:b],batch_size=32,path=IMG_BASE)

        image_embeddings=model.predict(curr_test_gen,verbose=1,use_multiprocessing=True,workers=4)
        image_embeds.append(image_embeddings)

    del model
    _=gc.collect()
    
    return np.concatenate(image_embeds)

# Function to generate embedding using arcface based model function
# the process is preformed in part due to RAM limitations
def embeddings_generation_arcface(model):
    image_embeds=[]
    for i,j in enumerate(range(C_LEN)):
        a=j*CHUNK
        b=min((j+1)*CHUNK,len(train))
        print('CHUNK:',a,"to",b)

        curr_test_gen=get_dataset(train.iloc[a:b].image_path.values,train.iloc[a:b].label_number.values)
        
        image_embeddings=model.predict(curr_test_gen,verbose=1,use_multiprocessing=True,workers=-1)
        image_embeds.append(image_embeddings)

    del model
    _=gc.collect()
    
    return np.concatenate(image_embeds)

# Prediction generation using KNN and Threshold value
def predictions_KNN(embeddings,neighbor_cnt,THRESHOLD_VALUE):
    model=NearestNeighbors(n_neighbors=neighbor_cnt)
    model.fit(embeddings)

    preds=[]
    IMAGE_THRESHOLD_DISTANCE=THRESHOLD_VALUE

    for j in range(C_LEN):
        a=j*CHUNK
        b=min((j+1)*CHUNK,len(train))
        #print('    CHUNK:',a,"to",b)

        distances,indices=model.kneighbors(embeddings[a:b,])

        for k in range(b-a):
            ID_SMALLER_DISTANCE=np.where(distances[k,]<IMAGE_THRESHOLD_DISTANCE)[0]
            ID_INDICES=indices[k,ID_SMALLER_DISTANCE]
            CURR_PREDS=train.iloc[ID_INDICES].posting_id.values
            preds.append(CURR_PREDS)

    del model,distances,indices
    _ = gc.collect()    
    
    return preds


# Prediction generation using Cosine Similarity and Threshold value
def predictions_cosine(embeddings,THRESHOLD_VALUE):
    cos_mat=cosine_similarity(embeddings,embeddings)
    cosine_threshold =THRESHOLD_VALUE
    mat=(cos_mat>cosine_threshold)

    cosine_predictions=[]
    for i in range(len(mat)):
        cosine_predictions.append(train[mat[i]].posting_id.values)

    cosine_predictions=pd.Series(cosine_predictions)
    
    del cos_mat,mat
    _=gc.collect()
    
    return cosine_predictions

# Normalize the embeddings to make the process of generating predictions easier
def normalize_embeddings(embeddings):
    for x in embeddings:
        norm = np.linalg.norm(x)
        x/=norm
    return embeddings


print("Image Embeddings using EfficientNetB4")
pretrained_model=EfficientNetB4(weights=None,include_top=False,input_shape=None)
model=create_model_arcface(pretrained_model)
model.load_weights('../input/shopee-model-training-tpu/best_model_efnb4.h5')
intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer("intermediate_output").output)

efficient_net_b4_embedding=embeddings_generation_arcface(intermediate_model)
efficient_net_b4_embedding=normalize_embeddings(efficient_net_b4_embedding)
np.save('efficient_net_b4_finetune_embedding.npy',efficient_net_b4_embedding)
print(f"Shape of EFFNETB4 embeddings:{efficient_net_b4_embedding.shape}")




print("Image Embeddings using ResNet101")
pretrained_model=ResNet101(weights=None,include_top=False,input_shape=None)
model=create_model_arcface(pretrained_model)
model.load_weights('../input/shopee-model-training-tpu/best_model_resnet101.h5')
intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer("intermediate_output").output)

resnet101_embedding=embeddings_generation_arcface(intermediate_model)
resnet101_embedding=normalize_embeddings(resnet101_embedding)
np.save('resnet101_embedding_finetune_embedding.npy',resnet101_embedding)
print(f"Shape of RESNET101 embeddings:{resnet101_embedding.shape}")




print("Image Embeddings using EfficientNetB5")
pretrained_model=EfficientNetB5(weights=None,include_top=False,input_shape=None)
model=create_model_arcface(pretrained_model)
model.load_weights('../input/shopee-model-training-tpu/best_model_efnb5.h5')
intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer("intermediate_output").output)


efficient_net_b5_embedding=embeddings_generation_arcface(intermediate_model)
efficient_net_b5_embedding=normalize_embeddings(efficient_net_b5_embedding)
np.save('efficient_net_b5_finetune_embedding.npy',efficient_net_b5_embedding)
print(f"Shape of EFFNETB5 embeddings:{efficient_net_b5_embedding.shape}")

