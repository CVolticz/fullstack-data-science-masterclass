# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion & Preprocessing
# MAGIC - Ingest Amazon Product Review Data
# MAGIC - Preprocessing Product Review Text 
# MAGIC - Generate Summary Statistics

# COMMAND ----------

# python import
import ast
import numpy as np
import pandas as pd

# pyspark import
import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

# Setup base directory
BASE_EX_DIR = spark.sql("DESCRIBE EXTERNAL LOCATION metastore_root_location")\
                .filter("name = 'metastore_root_location'")\
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
# MAGIC From initial glance at the data, it seem like the weight do not follow one unit system. Noticed sometime the weight is measured in gram and sometime the weight is measures in oz. While consider the project scope, these metrics might not be necessary. However, it might be worth it to see just how many records and what type of conversion we might need to do. Let's create some bar plot for our purpose

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
# MAGIC ## Processing Price Dictionary
# MAGIC Notice that the prices column is a little odd, the first investigation into the price column shows that the column contains a list of price dictionary object. Each dictionary object contains a price source in different currency unit, merchant, shipping type, and also date posted. This column is a perfect candidate for data mining. We will apply some techniques to isolate all information in this data properly and save it for future referenes.

# COMMAND ----------

# test viewing 
ast.literal_eval(product_review_raw_df['prices'].values[0].strip('][').split(', ')[0])

# COMMAND ----------

def isolate_dictionary(row):
    if row != row:
        return np.Nan

# COMMAND ----------

# parse pricing
# 1. turn string of list to just a list
product_review_raw_df['prices'] = product_review_raw_df['prices'].apply(lambda x: ast.literal_eval(x.strip('][').split(', ')))
product_review_raw_df = product_review_raw_df.assign(maxPrice=lambda x: x.prices[0]['amountMax'])\
                                            .assign(minPrice=lambda x: x.prices[0]['amountMin'])\
                                            .assign(currentPrice=lambda x: x.prices[0]['currentPrice'])

# COMMAND ----------


