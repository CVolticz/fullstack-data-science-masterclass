# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Analysis
# MAGIC EDA is perhaps the most important step in a data science process. Essentially, a good data scientist must be able to tell a complete story with the data before start building any model to answer a question. Additionally, EDA serves as a stepping stone to frame and reframe the ask from the business stakeholders, allows for natural conversation, and enable data scientists to provide guidance on business direction even before any advance modeling works. 
# MAGIC
# MAGIC Ofcourse, any analysis would requires data visualization tools. There are a few ways (internally and externally) to represent the story of your data as well as tailoring to the target audience. Here we'll first explore the python matplotlib/seborn library to perform some internal plots then switch these analysis over to reporting tools such as powerBI.

# COMMAND ----------

!pip install tqdm
!pip install langdetect
!pip install nltk
!pip install gensim==4.1.2

# COMMAND ----------

# python imports
import pandas as pd
import numpy as np

# NLP imports
from langdetect import detect
import gensim.downloader

# COMMAND ----------

# Load data
product_review_df = pd.read_csv("/Volumes/main/default/processed/product_review_data.csv")
product_review_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tokenization
# MAGIC Naturally, customers would form complete sentences when writting reviews. A complete sentence contains both the subject and predicates and convey a complete idea. Before we can teach a machine to form a complete sentence, we must break down the sentence to individual words through a process called Tokenization. While performing tokenization for given a complete message, a customer review in our case, it must be noted that there are a few words that are repeated often and provide no additional meanings. These are called stopwords and could therefore be removed. While performing the tokenization process, we can also create a vocabulatry containing all unique words along with an associated word index. This vocabulary can be use to transform words to numbers and vice versa at later stages. Let's perform tokenization on all of the custoemr reviews data by running the following code

# COMMAND ----------

# Count the number of words for each review
def word_count(data):
    """
        Function to clean the input dataset
        -----------------------------------
        Parameters: 
            - data : {pd.DataFrame(), input data}
            
        Returns:
            - data : {pd.DataFrame(), cleaned output data}
    """
    df_copy = data.copy()

    #take out observations that have NaNs
    subset = df_copy[['id', 'customerReviews']]
    df_copy.dropna(subset=['id', 'customerReviews'], inplace=True)
    
    #tokenize the review and get word count
    df_copy['review_tokenized'] = df_copy['customerReviews'].str.split()
    df_copy['word_num'] = df_copy['customerReviews'].str.split().str.len()
        
    #remove entries where only 5 or less words review
    df_copy['word_num'].astype('int32')
    df_copy = df_copy[df_copy.word_num > 1]
    
    return df_copy

preprocessed_reviews = word_count(product_review_df)
preprocessed_reviews[['word_num']].describe()

# COMMAND ----------

# Detecting review language
def language_detector(string):
    global i
    try:
        res = detect(string)
    except:
        res = "undetectable"
    if i % 1000 == 0:
        print(i)
    i = i + 1
    return res

i = 0
preprocessed_reviews['language'] = preprocessed_reviews['customerReviews'].apply(language_detector)
preprocessed_reviews

# COMMAND ----------

# Noted that majority of the reivews are under 500 words
# Taking a Logarithm scale there are also very few reviews less than 10 words.
# We can target our analysis to reviews above 10 words while truncating also the review length to 315 words
# NOTE: the tallest peak of our data is at around the 415 words point
preprocessed_reviews_spark = spark.createDataFrame(preprocessed_reviews)
display(preprocessed_reviews_spark)

# COMMAND ----------

# seem like there are a few instances where the review are written in another language.
# Since the size of the data in other is small, we can simply drop them
preprocessed_reviews_spark.groupBy("language").count().display()

# COMMAND ----------

preprocessed_reviews = preprocessed_reviews[preprocessed_reviews['language'] == 'en']
preprocessed_reviews

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Embeddings
# MAGIC
# MAGIC How should we think about embeddings relative to language? How do they represent words? Are they like dictionary definitions of words with clear boundaries?  Are they a sharp clear respresentation of the meaning or are they more nebulous?
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Sentimental Analysis
# MAGIC The dataset that were processed are product reviews obtained from Amazon. As such, the review data are natural language and thus captured the perspective, intention, and emotion of the customers. Useful information can be extract from textual data. However, computer don't quite understand textual data like you and I. On the other hand, computers are really good at performing calculation. So, in order to start extracting sentimental information from textual data we must transform textual data into numerial values and then have the computer learn to associate thes numbers with one another, creating contextual information in the process.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cosine Similarity
# MAGIC To measure the similarity of two words, we'll use the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between their representation vectors:
# MAGIC
# MAGIC $$ D^{cos}_{ij} = \frac{v_i^T v_j}{||v_i||\ ||v_j||}$$
# MAGIC
# MAGIC *Note that this is called cosine similarity because \\(D^{cos}\_{ij} = \cos(\theta_{ij})\\) , where \\(\theta_{ij}\\) is the angle between the two vectors.*
# MAGIC
# MAGIC
# MAGIC Let's use numpy to calculate the cosine similarity between two vectors.  The closer the distance between the two vectors the more alike they are.  In word2vec the embeddings are built so that words that are used in the same context have more similar vectors than words used in different contexts.  If you think of names of cities like London or Paris they will be used in the same context like "I want to visit ..." or "I flew in from ..." or "I once lived in ..."

# COMMAND ----------


