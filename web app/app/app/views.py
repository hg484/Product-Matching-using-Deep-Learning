from re import L
from app import app
from flask import render_template,request,redirect,session,url_for,json,jsonify
import os

# from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import gc,math,pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from app import download_data
from app import prepare_model

HEIGHT,WIDTH=256,256
CHANNELS=3

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["IMAGE_UPLOADS"] = "./app/static/img/upload_images/"
# app.config["DATASET_IMAGES"]="/static/img/train_images/"
app.config["DATASET_IMAGES"]="https://res.cloudinary.com/harshgupta/image/upload/shopee-images-cloudinary/train_images/"

# Removing uploaded files
for upload_file_name in os.listdir(app.config["IMAGE_UPLOADS"]):
    curr_file_path=app.config["IMAGE_UPLOADS"]+upload_file_name
    ext=curr_file_path.split('.')[-1]
    # print(ext)
    if ext.upper() not in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        continue
    os.remove(curr_file_path)

image_model=prepare_model.get_image_model()
text_model=prepare_model.get_text_model()

IMAGE_EMBEDDINGS_PATH="./app/model and embeddings/"
TEXT_EMBEDDINGS_PATH="./app/model and embeddings/"
efnb5_embeddings=prepare_model.normalize_embeddings(np.load(IMAGE_EMBEDDINGS_PATH+"efficient_net_b5_finetune_embedding.npy"))
sentence_transformer_embeddings=prepare_model.normalize_embeddings(np.load(TEXT_EMBEDDINGS_PATH+"sentence_transformer_stsb_mpnet_base_v2_embeddings.npy"))
merged_embeddings=np.concatenate([efnb5_embeddings,sentence_transformer_embeddings],axis=1)

train=pd.read_csv("./app/model and embeddings/train.csv")


def allowed_image(filename):
    
    if not "." in filename:
        return False

    ext=filename.rsplit(".",1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False



@app.route("/",methods={'GET','POST'})
def index():

    print(os.getcwd)

    if request.method=="POST":
        
        text=None
        image=None
        image_path=None
        random_text_data=None

        if request.form["text"]:
            missing=[]

            for k,v in request.form.items():
                if v=="":
                    missing.append(k)

            for k,v in request.files.items():
                if v.filename==""  :
                    missing.append(k)

            text=request.form.get("text")
            image=request.files.get("image")
            
            print(text)
            print(image)

            image_path=os.path.join(app.config["IMAGE_UPLOADS"],image.filename)

            if  missing:
                feedback=f"Missing fields for {', '.join(missing)}"
                return redirect(url_for("missing_data",feedback=feedback))
            
            if allowed_image(image.filename):
                image.save(os.path.join(app.config["IMAGE_UPLOADS"],image.filename))
                print("Image saved")
                # save image
            else:
                return redirect(url_for("wrong_filetype",allowed=False))
        
        elif request.form.get("demo_1_button"):
            print(1)
            text="Natur Hair Tonic Ginseng dan Aloe Vera 90ml"
            image_path="./app/static/img/demo_images/demo_image_3.jpg"

        elif request.form.get("demo_2_button"):
            print(2)
            text="SENKA Perfect Whip Facial Foam (Shiseido)"
            image_path="./app/static/img/demo_images/demo_image_perfect_whip.jpg"

        elif request.form.get("demo_3_button"):
            print(3)
            text="RATU MAYCREATE MOISTURIZING SPRAY - LOTION SPRAY MAY CREATE 150ML ORIGINAL"
            image_path="./app/static/img/demo_images/demo_image_lotion_spray.jpg"

        elif request.form.get("demo_4_button"):
            print(4)
            text="MISSHA Airy Fit Sheet Mask"
            image_path="./app/static/img/demo_images/demo_image_face_pack.jpg"

        elif request.form.get("demo_5_button"):
            print(5)
            text="Emina Glossy Stain" 
            image_path="./app/static/img/demo_images/demo_image_emina_glossy.jpg"
        
        elif  request.form.get("demo_6_button"):
            print("Random Input 1")
            text="Player wearing a hoodie"
            image_path="./app/static/img/demo_images/basketball player.jpg"

        elif  request.form.get("demo_7_button"):
            print("Random Input 2")
            text="Prime minster of Japan wearing a Suit"
            image_path="./app/static/img/demo_images/abe sad.jpg"
           


        text=[text]
        ds=prepare_model.get_dataset(image_path)
        image_embedding=prepare_model.normalize_embeddings(image_model.predict(ds))
        text_embedding=prepare_model.normalize_embeddings(text_model.encode(text))
        curr_embedding=np.concatenate([image_embedding,text_embedding],axis=1)

        print("embeddings generated")
        print(image_embedding.shape,text_embedding.shape)
        print(curr_embedding.shape)

        merged_preds=prepare_model.predictions_cosine_numpy(merged_embeddings,curr_embedding)
        # print(merged_preds)

        data=merged_preds.iloc[:15].reset_index()
        # print(data)

        # print(data['values'])

        # original code        similar_data=data[data["values"]<1.8]
        # similar_data=data[data["values"]<1.9]
        # similar_data=similar_data[similar_data["values"]>0.7]
        similar_data=data[data["values"]>0.7]
        similar_data=similar_data[similar_data["values"]<=1.8]

        identical_data=data[data['values']>=1.9]

        # similar_data=shuffle(similar_data,random_state=0)

        no_of_related_products=len(similar_data)
        related_image_paths=app.config["DATASET_IMAGES"]+similar_data.image.values
        related_text=similar_data.title.values

        no_of_identical_products=len(identical_data)
        identical_image_paths=app.config["DATASET_IMAGES"]+identical_data.image.values
        identical_text=identical_data.title.values


        print(data.shape)

        print("\nSimilar Data")
        print(len(related_image_paths))
        print(len(related_text))
        # print(related_text)


        original_img_path=image_path
        return redirect(url_for("result",original_img_path=original_img_path,original_text=text,related_text=related_text,related_image_paths=related_image_paths))


    return render_template('/public/index.html')

@app.route("/result")
def result():
    original_img_path=request.args["original_img_path"]
    original_text=request.args["original_text"]
    related_text=request.args['related_text']
    related_image_paths=request.args["related_image_paths"]
    # identical_text=request.args["identical_text"]
    # identical_image_paths=request.args["identical_image_paths"]
    # no_of_identical_products=request.args["no_of_identical_products"]
    # no_of_related_products=request.args["no_of_related_products"]


    return render_template("public/result.html",original_img_path=original_img_path,original_text=original_text,related_text=related_text,related_image_paths=related_image_paths)

@app.route("/trial_1")
def trial_1():
    return render_template("/public/index.html")


@app.route("/missing_data")
def missing_data():
    feedback=request.args["feedback"]
    return render_template("public/index.html",feedback=feedback)

@app.route("/wrong_filetype")
def wrong_filetype():
    allowed=request.args["allowed"]
    return render_template("/public/index.html",allowed=allowed)
