# Library used in the script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2,gc,math,os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# For this project i am using the RAPIDS Library developed by NVIDIA
# The RAPIDS suite of open source software libraries and APIs gives you the ability 
# to execute end-to-end data science and analytics pipelines entirely on GPUs
# more information: https://rapids.ai/
import cudf,cuml,cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors


# Limit tensorflow memory usage to 1GB leaving rest for RAPIDS
# RAPIDS Library requires GPU for extrememly fast computations when compared to numpy 
LIMIT=0
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

# Normalize the embeddings to make the process of generating predictions easier
def normalize_embeddings(embeddings):
    for x in embeddings:
        norm = np.linalg.norm(x)
        x/=norm
    return embeddings

# metric for measuring accuracy of pipeline
# F1 score for each image/text datapoint
# For the final F1 score average value is calculated
def get_metric_score(col1,col2):
    def f1score(row1,row2):
        n = len( np.intersect1d(row1,row2) )
        return 2*n / (len(row1)+len(row2))
    score=[]
    for i in range(len(col1)):
        row1=col1[i]
        row2=col2[i]
        score.append(f1score(row1,row2))
    
    return np.mean(score)

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

# Prediction generation using RAPIDS Library,Consine similarity and a set Threshold value
def predictions_cosine_rapids(embeddings,THRESHOLD_VALUE):
    
    # convert numpy to cupy type
    embeddings=cupy.array(embeddings)
    
    preds=[]
    THRESHOLD_DISTANCE=THRESHOLD_VALUE

    for j in range(C_LEN):
        a=j*CHUNK
        b=min((j+1)*CHUNK,len(train))
        # print('    CHUNK:',a,"to",b)
        
        matrix=cupy.matmul(embeddings,embeddings[a:b].T).T
                
        for k in range(b-a):
            ID=np.where(matrix[k,]>THRESHOLD_DISTANCE)[0]
            CURR_PREDS=train.iloc[cupy.asnumpy(ID)].posting_id.values
            preds.append(CURR_PREDS)

    del matrix,embeddings
    _ = gc.collect()    
    
    return preds



# Prediction generation using Numpy Library,Consine similarity and a set Threshold value
def predictions_cosine_numpy(embeddings,THRESHOLD_VALUE):
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



# Data Path and image base path 
DATA_PATH='../input/shopee-product-matching/'
IMG_BASE=DATA_PATH+'train_images/'

# Data Prepartation and target generation
train=pd.read_csv(DATA_PATH+"train.csv")

trgt=train.groupby("label_group").posting_id.agg("unique")
train['target']=train.label_group.map(trgt)

train['image_path']=DATA_PATH+'train_images/'+train.image

id_to_label_mapping=dict(zip(train.label_group.unique(),range(train.label_group.nunique())))
train["label_number"]=train.label_group.map(id_to_label_mapping)

NUM_CLASSES=train.label_group.nunique()

CHUNK=1024*1
C_LEN=len(train)//CHUNK
if len(train)%CHUNK!=0: C_LEN += 1
    
# Base path for embeddings storage
BASE_PATH="../input/"
IMAGE_EMBEDDINGS_PATH=BASE_PATH+"product-matching-image/"
TEXT_EMBEDDINGS_PATH=BASE_PATH+"shopee-text-new/"

# loading previously created embeddings from .npy files 
efnb4_embeddings=np.load(IMAGE_EMBEDDINGS_PATH+"efficient_net_b4_finetune_embedding.npy")
efnb5_embeddings=np.load(IMAGE_EMBEDDINGS_PATH+"efficient_net_b5_finetune_embedding.npy")
resnet101_embeddings=np.load(IMAGE_EMBEDDINGS_PATH+"resnet101_embedding_finetune_embedding.npy")
tfidf_embeddings=np.load(TEXT_EMBEDDINGS_PATH+"tfidf_text_embeddings.npy")
sentence_transformer_embeddings=np.load(TEXT_EMBEDDINGS_PATH+"sentence_transformer_stsb_mpnet_base_v2_embeddings.npy")

# Embedding normalization
efnb4_embeddings=normalize_embeddings(efnb4_embeddings)
efnb5_embeddings=normalize_embeddings(efnb5_embeddings)
resnet101_embeddings=normalize_embeddings(resnet101_embeddings)
tfidf_embeddings=normalize_embeddings(tfidf_embeddings)
sentence_transformer_embeddings=normalize_embeddings(sentence_transformer_embeddings)

# Shape of the embeddings being used
print(f"EFNB4 Embedding shape {efnb4_embeddings.shape}")
print(f"EFNB5 Embedding shape {efnb5_embeddings.shape}")
print(f"RESNET101 Embedding shape {resnet101_embeddings.shape}")
print(f"TFIDF Embedding shape {tfidf_embeddings.shape}")
print(f"SENTENCE TRANSFORMER Embedding shape {sentence_transformer_embeddings.shape}")

# Baseline using phash 
print("\n")
IMG_PHASH=train.groupby('image_phash').posting_id.agg('unique').to_dict()
train['phash']=train.image_phash.map(IMG_PHASH)
print(f"Score for phash baseline : {round(get_metric_score(train.target,train.phash),4)}")

# Performance of Sentence Transformers embeddings
print('\n')
THRESHOLD_VALUE=0.75
preds=train["st_preds"]=predictions_cosine_rapids(sentence_transformer_embeddings,THRESHOLD_VALUE)
print("Sentence Transformers embeddings only:")
print("THRESHOLD=",THRESHOLD_VALUE,"\nScore:",round(get_metric_score(train.target,preds),4))

# Performance of TfidfVectorizer embeddings
print('\n')
THRESHOLD_VALUE=0.65
preds=train["tfidf_preds"]=predictions_cosine_rapids(tfidf_embeddings,THRESHOLD_VALUE)
print("TFIDF embeddings only:")
print("THRESHOLD=",THRESHOLD_VALUE,"\nScore:",round(get_metric_score(train.target,preds),4))

# Performance of ResNet101 embeddings
print('\n')
THRESHOLD_VALUE=0.85
preds=train["res101_preds"]=predictions_cosine_rapids(resnet101_embeddings,THRESHOLD_VALUE)
print("Resnet101 embeddings only:")
print("THRESHOLD=",THRESHOLD_VALUE,"\nScore:",round(get_metric_score(train.target,preds),4))

# Performance of EfficientNetB4 embeddings
print('\n')
THRESHOLD_VALUE=0.85
preds=train["efnb4_preds"]=predictions_cosine_rapids(efnb4_embeddings,THRESHOLD_VALUE)
print("EfficientNetB4 embeddings only")
print("THRESHOLD=",THRESHOLD_VALUE,"\nScore:",round(get_metric_score(train.target,preds),4))

# Performance of EfficientNetB5 embeddings
print('\n')
THRESHOLD_VALUE=0.85
preds=train["efnb5_preds"]=predictions_cosine_rapids(efnb5_embeddings,THRESHOLD_VALUE)
print("EfficientNetB5 embeddings only:")
print("THRESHOLD=",THRESHOLD_VALUE,"\nScore:",round(get_metric_score(train.target,preds),4),"\n")

#Merging Process used to generate final predicitions
#First Approach: Embedding Merging
print('\n')
THRESHOLD_VALUE=0.55
preds=train["merge_preds_first"]=predictions_cosine_rapids(normalize_embeddings(np.concatenate([efnb4_embeddings,resnet101_embeddings,tfidf_embeddings,sentence_transformer_embeddings],axis=1)),THRESHOLD_VALUE)
print("First Approach: Merging embeddings of EfficientNetB4,ResNet101,TFIDF and Sentence Transformers")
print("Score:",round(get_metric_score(train.target,preds),4)," THRESHOLD=",THRESHOLD_VALUE)

#Second Approach: Prediction Merging
print('\n')
merge_columns=["tfidf_preds","st_preds","res101_preds","efnb4_preds"]
for col in merge_columns[1:]:
    train['merge_preds_second'] = train[merge_columns[0]].apply(lambda x: x.tolist()) + train[col].apply(lambda x: x.tolist())
train.merge_preds_second=train.merge_preds_second.apply(lambda x:np.unique(x))
print("Second Approach: Merging predictions of EfficientNetB4,ResNet101,TFIDF and Sentence Transformers")
print("Score:",round(get_metric_score(train.target,train.merge_preds_second),4),"\n\n\n")

# Save predictions
train.to_csv("final_predictions.csv",index=False)
