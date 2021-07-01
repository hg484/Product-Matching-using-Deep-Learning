# Product Matching for E-Commerce using Deep Learning 

E-commerce has seen a increadible surge in users over the past few years.Thus for each E-commerce company it is has become increabily important to provide high quality search results and recommendations.With millions of third party sellers operating on their websites ,the process of distinguishing  between products have become increasing difficult

**Thus the Goal of this project is to develop an efficient stratergy to find similar products avaliable by utilizing the product's image and text label.**


### Why don't we just compare image and text directly?
Each image cannot be compared one by one with the whole image dataset,This approach will be increadibly expensive computationally and time intensive in nature due to sheer size of images.Also process of comparing texts still remains difficult problem 

**Hence fintuned pretrained CNN models will be used to generate image embeddings ,and a similar approach is utilized to convert text data into word embeddings using TfidfVectorizer and a Transformer, this approach produced an average F1 score of 0.87 when compared to baseline score of 0.55**


### What are Embeddings
Embeddings are vector representation of data formed by converting high dimesional data (Image, text ,sound files etc) into relatively low dimensional tabular data.They make it easier to perform machine learning on large inputs.
[More Information](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture#:~:text=An%20embedding%20is%20a%20relatively,like%20sparse%20vectors%20representing%20words.&text=An%20embedding%20can%20be%20learned%20and%20reused%20across%20models.)

This dataset was provided by Shopee ,Shopee is s a Singaporean multinational technology company which focuses mainly on e-commerce.


## Approach Utilized
## Image based stratergy
Rather than creating our own model for embedding generation, the best method is to use state of the art image models then finetune them on our dataset.Using these pretrained model without any fintuning will provide an average result(average F1 score of 0.59) whereas the finetuned model performs much better(average F1 score of 0.73).Below image represents the model used generate image embeddings.
 <br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/shopee%20Image%20model.png" alt="Image Model"> </p>
 <br/>

The process of model finetuing is borrowed from facial recognition system, [ArcFace Margin Layer](https://arxiv.org/abs/1801.07698) is used instead of a softmax layer in the model during  finetuning process.

#### Advantage of ArcFace Layer
Unlike Softmax,it explicitly optimizes feature embeddings to enforce higher similarity between same class data , this inturn leads to higher quality of embeddings being generated.
<br />
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/ArcFace%20vs%20Softmax.png" alt="Softmax vs Arcface Margin"> </p>
 <br/>

After embeddings generation, the goal is to generate accurate predicitions using KNearestNeighbour Algorithm and Cosine Similarity. Due to the large number of input data, the sklearn framework cannot be utlized as it leads to **Out of Memory Error**.Hence the [RAPIDS](https://rapids.ai/index.html) library is used, It is a opensource framework used to accelerate data science process by providing ability to execute end-to-end data science and analytics pipelines entirely on GPUs.
 <br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/Image%20Prediction.png"  alt="Image Predictions"> </p>
 <br/>

Predictions from all the different image models are merged together to generate the final image based predictions.


## Text based stratergy
Product's text label are converted into word embeddings using two different approaches, TfidfVectorizer and Sentence Transformer are used to encode every text label.
<br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/shopee%20text%20model.png" alt="Text Model"> </p>
 <br/>
 
#### TfidfVectorizer
[TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/#:~:text=TF%2DIDF%20is%20a%20statistical,across%20a%20set%20of%20documents.) (term frequency–inverse document frequency) is used to under stand the relevance of a word present in the document,[TfidfVectoriver](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a) utilizes the tfidf values to generate word embeddings for each and every label

#### Sentence Transformers
[Sentence Transformers](https://github.com/UKPLab/sentence-transformers) is a framework which provides an easy method to generate vector representations of text by utilizing transformer networks like BERT,RoBERTa etc,for this application a pretrained transformer is used to generate sentence embeddings for finding the semantic similarity between text data.


<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/text%20predictions.png"  alt="Text Predictions"> </p>

<br/>
After embedding generation, both the tfidf and transformer embeddings are merged to into a single set of embeddigs and the final prediction are calculated

## Prediction generation using  Cosine Similarity and KNearestNeigbour Algorithm + Merging Approach
#### Cosine Similarity
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/cosine_similarity_2.png" alt="cosine similarity"></p>
Cosine similarity tells us the similarity between two different vectors by calculating the cosine of the angle between two vector and determines whether the two vectors lie in the same directions, to generate the final predictions a minimum threshold distance is decided and all datapoints with a similarity value greater than the threshold value are the required predictions. (Higher the similarity value, Closer the relation between datapoints)

#### KNearestNeighbour Algorithm
NearestNeighbour is a common algorithm used find the required number of nearest datapoint according a chosen metric,this allows us to find acurate predictions by deciding a minimum threshold distance and all data with distance less than the decided threshold will be the required predictions .(Lower the distance, Closer the relation between datapoints)
 
#### Merging Approach
##### First Approach 
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/prediction%20approach%201.png" alt="1st appoach">
</p>
 
##### Second Approach 
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/prediction%20approach%202.png" alt="2nd approach">
</p>

First Approach performs slightly better than the second approach,a possible reason for this is  1st approach allows the predictions to be developed using both image and text embeddings rather than just merging indepents predictions
 
Implementations of both the predicition methods are performed using the open source library developed by NIVIDIA called RAPIDS

## Results
##### Metric Used and how its calculated:
Metric used to judge the performace is Average F1 Score, for each data entry F1 score is calculated and then mean of all F1 Score are taken.F1 score measures a test's accuracy, it is calculated using precision and recall of the test
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/f1_score.svg" alt="F1 Score">
</p>

##### Setting baseline using pHash

Used dataset provides the predictions using [pHash](https://www.phash.org), pHash is a fingerprint of a multimedia file derived from various features,If pHash are 'close' enough ,then datapoint are similar
##### Baseline Average F1 Score:0.55
##### Image Only Score(Fintuned CNN):0.72
##### Text Only Score: 0.62
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/result%20new%201.png" alt="Resuts">
</p>


### Final Merged Score(Image+Text): 0.87
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/result%20new%202.png" alt="Resuts">
</p>



