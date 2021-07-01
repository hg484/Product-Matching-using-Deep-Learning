import numpy as np
import pandas as pd
import gc,math,pickle,os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_model_filename="./app/model and embeddings/best_model_efnb5.h5"
text_model_filename="./app/model and embeddings/sentence_transformers_model.sav"

data=pd.read_csv("./app/model and embeddings/train.csv")


HEIGHT,WIDTH=256,256
CHANNELS=3
BATCH_SIZE=32
NUM_CLASSES=11014

def normalize_embeddings(embeddings):
    for x in embeddings:
        norm = np.linalg.norm(x)
        x/=norm
    return embeddings

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
    x = tf.keras.layers.GlobalAveragePooling2D(name="intermediate_output")(x)
    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype='float32')(x)

    model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
    
    return model


def get_image_model():
    pretrained_model=EfficientNetB5(weights=None,include_top=False,input_shape=None)
    model=create_model_arcface(pretrained_model)
    model.load_weights(image_model_filename)
    intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer("intermediate_output").output)
    return intermediate_model


def get_text_model():
    text_model=pickle.load(open(text_model_filename,"rb"))
    return text_model

def arcface_format(image,label):
    return {'image_input':image,'label_input':label},label


def process_data(image_path,label):
    img=tf.io.read_file(image_path)
    img=tf.image.decode_jpeg(img,channels=CHANNELS)
    img=tf.image.resize(img,[HEIGHT,WIDTH])
    # img = tf.cast(img, tf.float32) / 255.0
    return img,label

def get_dataset(image):
    temp_label=pd.Series(-1).values
    filepaths =pd.Series(image).values

    ds=tf.data.Dataset.from_tensor_slices((filepaths,temp_label))
    ds=ds.map(process_data)
    ds=ds.map(arcface_format)
    ds=ds.batch(1)
    return ds

def predictions_cosine_numpy(embeddings,curr_embedding):
    cos_mat=np.matmul(embeddings,curr_embedding.T).T
    THRESHOLD_VALUE=0.0
    cosine_threshold =THRESHOLD_VALUE
    mat=(cos_mat>cosine_threshold)
    mat=np.reshape(mat,(len(embeddings),))

    # print(np.min(cos_mat),np.max(cos_mat),np.sum(cos_mat),np.sum(mat))
    # print(data[mat])
    data['values']=np.reshape(cos_mat,(len(embeddings),))
    return data[mat].sort_values(by='values',ascending=False)