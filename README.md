# Product Matching for E-Commerce using Deep Learning 

### Deployment stopped due to GCP charges

Not Working Currently: Deployed Web App : [http://product-matching-webapp.el.r.appspot.com/](http://product-matching-webapp.el.r.appspot.com/)
### Youtube Link:[https://www.youtube.com/watch?v=uQq281Uzb9k](https://www.youtube.com/watch?v=uQq281Uzb9k)


E-commerce has seen an incredible surge in terms of the users over the past few years. The transition to E-commerce has been further accelerated by the COVID-19 pandemic. Thus for each E-commerce companies, it is has become increasingly important to provide high-quality search results and recommendations. With millions of third-party sellers operating on their websites, the process of distinguishing between products have become increasingly difficult.

**The Goal of this project is to develop an efficient strategy to find similar products available on an e-commerce website by utilizing the product's image and text label.**

<p align="center" > 
  <img src="https://github.com/harsh-miv/Product-Matching-Web-App-Media-FIles/blob/main/productmatching-screen%20share%20higher%20quality.gif" />
</p>


### Few Examples:
#### Example 1:
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_1_1.png" alt="example 1.1"> </p>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_1_2.png" alt="example 1.2"> </p>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_1_3.png" alt="example 1.3"> </p>

#### Example 2:
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_2_1.png" alt="example 2.1"> </p>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_2_2.png" alt="example 2.2"> </p>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/example_2_3.png" alt="example 2.3"> </p>

### Structure of Web App
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/product%20matching%20deployemnt%20structure%20v2.png" alt="Web app structure"> </p>


### Why don't we just compare image and text directly?
Each image cannot be compared one by one with the whole image dataset. This approach will be incredibly computational expensive and excessively time-intensive in nature due to the sheer size of images. The process of comparing texts directly also may not give the desired outcomes.

**Hence a fine-tuned pre-trained CNN models can be used to generate image embeddings and a similar approach is utilized to convert text data into word embeddings using TfidfVectorizer and a Transformer. This approach produces an average F1 score of 0.87 when compared to a baseline score of 0.55.**


### What are Embeddings
Embeddings are a vector representation of data formed by converting high dimensional data (Image, text, sound files etc.) into relatively low dimensional tabular data. They make it easier to perform machine learning on large inputs.[More Information](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture#:~:text=An%20embedding%20is%20a%20relatively,like%20sparse%20vectors%20representing%20words.&text=An%20embedding%20can%20be%20learned%20and%20reused%20across%20models.)

This dataset was provided by Shopee ,Shopee is s a Singaporean multinational technology company which focuses mainly on e-commerce.


## Approach Utilized
## Image-based strategy
Rather than creating our model for embedding generation, the best method is to use state of the art image models then fine-tune them on our dataset. Using these pre-trained models without any fine-tuning will provide an average result (average F1 score of 0.59) whereas the fine-tuned model performs much better (average F1 score of 0.73). The image below represents the model used to generate image embeddings.
 <br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/shopee%20Image%20model.png" alt="Image Model"> </p>
 <br/>

The process of model fine-tuning is borrowed from the facial recognition system, [ArcFace Margin Layer](https://arxiv.org/abs/1801.07698) is used instead of a softmax layer in the model during the fine-tuning process.


#### Advantage of ArcFace Layer
Unlike Softmax, it explicitly optimizes feature embeddings to enforce higher similarity between same class data, this, in turn, leads to a higher quality of embeddings being generated.
<br />
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/ArcFace%20vs%20Softmax.png" alt="Softmax vs Arcface Margin"> </p>
 <br/>

After embeddings generation, the goal is to generate accurate predictions using KNearestNeighbour Algorithm and Cosine Similarity. Due to a large number of input data, the sklearn framework cannot be utilized as it leads to **Out of Memory Error**.Hence the [RAPIDS](https://rapids.ai/index.html)  library is used, it is an open-source framework used to accelerate the data science process by providing the ability to execute end-to-end data science and analytics pipelines entirely on GPUs.
 <br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/Image%20Prediction.png"  alt="Image Predictions"> </p>
 <br/>

Predictions from all the different image models are merged by utilizing either of the prediction approaches to be discussed later in the document to generate the final image based predictions.


## Text-based strategy
The product’s text label is converted into word embeddings using two different approaches, TfidfVectorizer and Sentence Transformer are used to encode every text label.
<br/>
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/shopee%20text%20model.png" alt="Text Model"> </p>
 <br/>
 
#### TfidfVectorizer
[TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/#:~:text=TF%2DIDF%20is%20a%20statistical,across%20a%20set%20of%20documents.) (term frequency–inverse document frequency) is used to under stand the relevance of a word present in the document,[TfidfVectoriver](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a) utilizes the tfidf values to generate word embeddings for each and every label

#### Sentence Transformers
[Sentence Transformers](https://github.com/UKPLab/sentence-transformers) is a framework which provides an easy method to generate vector representations of text by utilizing transformer networks like BERT,RoBERTa etc,for this application a pretrained transformer is used to generate sentence embeddings for finding the semantic similarity between text data.


<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/text%20predictions.png"  alt="Text Predictions"> </p>

<br/>
After embedding generation, both the tfidf and transformer embeddings can be used by either of the prediction approaches for final prediction calculation.
This dataset was provided by Shopee from their indonesian division for a data science competition, Shopee is s a Singaporean multinational technology company that focuses mainly on e-commerce.


## Prediction generation using Cosine Similarity and KNearestNeigbour Algorithm + Merging Approach
#### Cosine Similarity
<p align="center"> <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/cosine_similarity_2.png" alt="cosine similarity"></p>
Cosine similarity tells us the similarity between two different vectors by calculating the cosine of the angle between two vectors and determines whether the two vectors lie in the same directions, to generate the final predictions a minimum threshold distance is decided and all data points with a similarity value greater than the threshold value are the required predictions. (Higher the similarity value, closer the relation between data points).

#### KNearestNeighbour Algorithm
NearestNeighbour is a common algorithm used to find the required number of nearest data points according to a chosen metric. This allows us to find accurate predictions by deciding a minimum threshold distance. All data points with a distance less than the decided threshold will be the required predictions. (Lower the distance, closer the relation between data points).
 
#### Merging Approach
##### First Approach 
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/prediction%20approach%201.png" alt="1st appoach">
</p>
 
##### Second Approach 
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/prediction%20approach%202.png" alt="2nd approach">
</p>

First Approach performs slightly better than the second approach. The 1st approach allows the predictions to be developed using the merged embeddings (both image and text embeddings) whereas the second approach uses merging independents predictions.
 
Implementations of both the prediction methods are performed using the open-source library developed by NVIDIA called RAPIDS.

## Results
##### Metric Used and how its calculated:
The Metric used to judge the performance is the Average F1 Score. For each data entry, the F1 score is calculated and then the mean of all F1 Scores is taken.F1 score measures a test's accuracy, it is calculated using precision and recall of the test.
<p align="center"> 
 <img src="https://github.com/harsh-miv/Product-Matching-using-Deep-Learning/blob/master/Diagrams%20and%20Images/f1_score.svg" alt="F1 Score">
</p>

##### Setting baseline using pHash

The used dataset provides the predictions using using [pHash](https://www.phash.org), pHash is a fingerprint of a multimedia file derived from various features, If pHash are 'close' enough, then datapoint is similar.
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



