# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion & Preprocessing
# MAGIC - Ingest Amazon Product Review Data
# MAGIC - Preprocessing Product Review Text 
# MAGIC - Generate Summary Statistics

# COMMAND ----------

!pip install tqdm

# COMMAND ----------

# python import
import ast
import numpy as np
import pandas as pd

# pyspark import
import pyspark.sql.functions as F
import pyspark.sql.types as T

# progress bar
from tqdm.notebook import tqdm

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTERNAL LOCATION dev_adls_uc;

# COMMAND ----------

# Setup base directory
BASE_EX_DIR = spark.sql("DESCRIBE EXTERNAL LOCATION dev_adls_uc")\
                .filter("name = 'dev_adls_uc'")\
                .select("url").head()[0]

# COMMAND ----------

# Load data from volume & perform a quick data profile
# Since this dataset is relatively small, we'll be able to load it in using traditional Pandas DataFrame
product_review_raw_df = pd.read_csv("/Volumes/main/default/raw/kaggle_amazon_product_reviews.csv")
product_review_raw_df.columns

# COMMAND ----------

# Isolate columns of interest
product_review_raw_df = product_review_raw_df[['id', 'categories', 
                                               'dateAdded', 'dateUpdated',
                                               'dimension', 'weight', 
                                               'manufacturer', 'name', 'prices',
                                               'reviews.date', 'reviews.numHelpful',
                                               'reviews.rating', 'reviews.title',
                                               'reviews.text', 'reviews.username']]
product_review_raw_df

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA
# MAGIC Performing some initial high level analysis of the dataset.

# COMMAND ----------

# 1. Let's get information of the dataframe (row numbers, amount of nulls, data typing, etc.)
product_review_raw_df.info()

# COMMAND ----------

# Interestingly, a lot of the columns read in as object or string type.
# This is not ideal for analysis, we'll need to make some changes
# Relevant columns that need to be processed are:
## dateAdded, dateUpdated, dimension, weight prices, reviews.date	

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Date Columns
# MAGIC Since the overall objective is to extract review information of the product, the actual time probably do not matter. For that, we'll format all of the date columns into a more readable 'YYYY-MM-DD' datetime format 

# COMMAND ----------

product_review_raw_df['dateAdded'] = pd.to_datetime(product_review_raw_df['dateAdded'], format='%Y-%m-%d %H:%M:%S')
product_review_raw_df['dateUpdated'] = pd.to_datetime(product_review_raw_df['dateUpdated'], format='%Y-%m-%d %H:%M:%S')
product_review_raw_df['reviews.date'] = pd.to_datetime(product_review_raw_df['reviews.date'], format='%Y-%m-%d %H:%M:%S')
product_review_raw_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Weight
# MAGIC From initial glance at the data, it seems like the weight do not follow one unit system. Noticed sometime the weight is measured in gram and sometime the weight is measures in oz. While consider the project scope, these metrics might not be necessary. However, it might be worth it to see just how many records and what type of conversion we might need to do. Let's create some bar plot for our purpose.

# COMMAND ----------

product_review_raw_df['weight'] = product_review_raw_df['weight'].str.split(" ")
product_review_raw_df

# COMMAND ----------

def extract_weight_units(row, index):
    # if row contains no value
    if row != row:
        return np.nan
    else:
        return row[index]

product_review_raw_df['units'] = product_review_raw_df['weight'].apply(lambda x: extract_weight_units(x, 1))
product_review_raw_df['weight_value'] = product_review_raw_df['weight'].apply(lambda x: extract_weight_units(x, 0))
product_review_raw_df['weight_value'] = product_review_raw_df['weight_value'].astype(float)

# COMMAND ----------

# convert to dataframe to spark to leverage databricks build in visualization
# Noted that we only have 3 types of metrics (lbs and punds means the same thing)
# Two of the metric, lbs and onnces, belong to the English system
# One of the metric, grams, belong to the Metric system
# Since lbs is the dominance metric type, let's convert everything to lbs
product_review_spark = spark.createDataFrame(product_review_raw_df)
display(product_review_spark)

# COMMAND ----------

# Convert Units To One Metric System
g_to_lbs = 0.00220462
oz_to_lbs = 0.0625
product_review_raw_df['weight_value'] = np.where(product_review_raw_df['units'] == 'ounces', 
                                                 product_review_raw_df['weight_value']*oz_to_lbs, 
                                                 np.where(product_review_raw_df['units'] == 'grams',
                                                        product_review_raw_df['weight_value']*g_to_lbs,
                                                        product_review_raw_df['weight_value'] ))
product_review_raw_df['units'] = 'lbs'
product_review_spark = spark.createDataFrame(product_review_raw_df)
display(product_review_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Renamming Columns and Write Processed Data

# COMMAND ----------

renaming_cols={
    "reviews.date": "reviewsDate",
    "reviews.numHelpful": "helpfullness",
    "reviews.rating": "rating",
    "reviews.title": "reviewsTitle",
    "reviews.text": "customerReviews",
    "reviews.username": "username"
}

product_review_raw_df = product_review_raw_df.rename(columns=renaming_cols)

# COMMAND ----------

# write data out
product_review_raw_df.to_csv("/Volumes/main/default/processed/product_review_data.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Price Dictionary
# MAGIC Notice that the prices column is a little odd, the first investigation into the price column shows that the column contains a list of price dictionary object. Each dictionary object contains a price source in different currency unit, merchant, shipping type, and also date posted. This column is a perfect candidate for data mining. We will apply some techniques to isolate all information in this data properly and save it for future referenes.

# COMMAND ----------

product_review_raw_df['prices'].values[0].strip('][').split(', ')

# COMMAND ----------

# test view of 1 row of data
ast.literal_eval(product_review_raw_df['prices'].values[0].strip('][').split(', ')[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price Dictionary - Design Thinking
# MAGIC Because of the nature of the price dictionary data, we would need to create a dataframe for each row of data. This technique is commonly known as internal data mining. Here we're essentially extract more information from a given piece of data. One thing to note while mining this data is be sure to also create a way for this piece of data to associate back to the original. Here's I'm thinking of a duplicated product id column with multiple rows of pricing information (min,max,currency units, etc.)

# COMMAND ----------


def isolate_dict(row):
    """
        given a row of string
        split it into a dictionary
    """
    if row != row:
        return np.nan
    else:
        return row.strip('][').split(', ')[0]

def create_dataframe(row):
    """
        Given a row contains a dictionary,
        returns the key: value pairs
    """
    if row != row:
        return np.nan
    
    # specified the pertinent data columns
    data_columns = ['amountMax',
                    'amountMin',
                    'condition',
                    'currency',
                    'dateAdded',
                    'dateSeen',
                    'isSale',
                    'merchant',
                    'shipping',
                    'sourceURLs']

    temp_df_list = []
    for price_dict in row:
        # if the dictionary doesn't have a column, create one and holds an NaN value
        for d in data_columns:
            if d not in price_dict.keys():
                price_dict[d] = np.nan 

        # create a data frame from the data
        df = pd.DataFrame(columns=data_columns).from_dict(price_dict, orient='index').T
        temp_df_list.append(df)

    return pd.concat(temp_df_list).reset_index(drop=True)

# COMMAND ----------

# parse pricing
# 1. turn string of list to just a list
product_review_raw_df['prices_dict'] = product_review_raw_df['prices'].apply(lambda x: ast.literal_eval(x))

# 2. for each row of pricing data return a data frame of the associated pricing
pricing_df_list = []
for i, row in tqdm(product_review_raw_df.iterrows()):
    # capture the row ID
    product_id = row['id'] 
    temp_price_df = create_dataframe(row['prices_dict'])
    try:
        # attach product id and add it to the general dataframe
        temp_price_df['id'] = product_id
        pricing_df_list.append(temp_price_df)
    except Exception as e:
        print(e)
        pass

pricing_df = pd.concat(pricing_df_list, axis=0).reset_index(drop=True)
pricing_df

# COMMAND ----------

# noted that two columns dateSeen and SourceURLs contains a list of multiple items. Let's expand the those data for additonal information
pricing_df = pricing_df.explode(["dateSeen"])
pricing_df = pricing_df.explode("sourceURLs")
pricing_df = pricing_df.drop_duplicates(keep='first')
pricing_df = pricing_df.reset_index(drop=True)
pricing_df

# COMMAND ----------

pricing_df.info()

# COMMAND ----------

# check to see if the ability data can be fill
# noted that since we're only seeing two In Stock and Yes, and there is no other column we can reference to. We cal also safely drop this column
# we ceratinly can build a scraper to get this information if necessary through the given URL.
availability_df = pricing_df[~(pricing_df['availability'] != pricing_df['availability'])][['availability', 'isSale']]
availability_df['availability'].unique(), pricing_df['isSale'].unique()

# COMMAND ----------

# a quick dataframe info shows that a few columns of data (condition, offer, returnPolicy, warranty) have very few row of data. Compared to the size of the dataframe, I think we can safely drop these columns
pricing_df = pricing_df.drop(columns=["condition", "offer", "returnPolicy", "warranty", "availability"])
pricing_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Pricing Out
# MAGIC From 1,597 rows of data, we were able to obtain about 20,100 extra rows of pricing information. Since this piece of information is valudation, it is worth writing it back out to our datalake for future analysis.

# COMMAND ----------

# write data out to ADLS
pricing_df.to_csv("/Volumes/main/default/processed/amazon_product_pricing_data.csv")

# COMMAND ----------


