# ! pip install cudf
# ! pip install cuml
# ! pip install cupy

import numpy as np
import pandas as pd
import gc,cv2,math
import matplotlib.pyplot as plt

# Deep learning framework : tensorflow & keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4,ResNet101,EfficientNetB5
from sklearn.metrics.pairwise import cosine_similarity

# Sentence transformers provide a frameworks to develop vector representations from text data
# More Information: https://github.com/UKPLab/sentence-transformers
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer

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

# metric for measuring accuracy of pipeline
# F1 score for each image/text datapoint is calculated
# and for the final F1 score average value of all datapoints are taken
def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

# values of embeddings lie between a large range
# normanlizing values makes the process easier
def normalize_embeddings(embeddings):
    for x in embeddings:
        norm = np.linalg.norm(x)
        x/=norm
    return embeddings


HEIGHT,WIDTH=256,256
CHANNELS=3
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=32


# Basic data preparations
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
train_gf = train_cu =cudf.DataFrame(train)

# print("Sample of available data")
# print(f"Train Data shape is {train.shape}")
# print(train.head())


# Setting a baseline using PHASH values
IMG_PHASH=train.groupby('image_phash').posting_id.agg('unique').to_dict()
train['phash']=train.image_phash.map(IMG_PHASH)

train["f1"]=train.apply(getMetric("phash"),axis=1)
print(f"F1 Score for phash baseline = {train.f1.mean()}")


# For more details regarding to selection of pretrained language model 
# More Information: 
#https://www.sbert.net/docs/usage/semantic_textual_similarity.html, https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('stsb-mpnet-base-v2')
sentence_embeddings = model.encode(train.title.values)
np.save('sentence_transformer_stsb_mpnet_base_v2_embeddings.npy',sentence_embeddings)
print(sentence_embeddings.shape)

# Tfidf vectorizer model, tfidf value for a word tells the importance of a word in a document
# more information: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
FEATURE_CNT=25000
model=TfidfVectorizer(binary=True,max_features=FEATURE_CNT)
tfidf_text_embeddings=model.fit_transform(train_cu.title)
tfidf_text_embeddings_numpy=cupy.asnumpy(tfidf_text_embeddings)
tfidf_text_embeddings=tfidf_text_embeddings.toarray()
np.save('tfidf_text_embeddings.npy',tfidf_text_embeddings)
print(tfidf_text_embeddings.shape)