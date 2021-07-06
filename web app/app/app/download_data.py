import os
from tqdm import tqdm
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="service-account-file-harsh-gupta.json"


from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob."""
 
    storage_client = storage.Client()
 
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
 
    print(
        "Blob {} downloaded to file path {}. successfully ".format(
            source_blob_name, destination_file_name
        )
    )

SAVE_PATH=os.getcwd()+"/app/model and embeddings/"
file=open("./app/model and embeddings/filenames_model_embeddings.txt","r")
file_names=file.read().split("\n")

for file in tqdm(os.listdir(SAVE_PATH)):
    curr_file_path=SAVE_PATH+file
    ext=curr_file_path.split('.')[-1]
    # print(ext)
    if ext=='txt':
        continue
    os.remove(curr_file_path)


for curr_file in tqdm(file_names):
    download_blob(bucket_name="embeddings-models-product-matching-webapp",source_blob_name=curr_file,destination_file_name=SAVE_PATH+curr_file)