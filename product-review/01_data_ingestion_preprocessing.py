# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion & Preprocessing
# MAGIC - Ingest Amazon Product Review Data
# MAGIC - Preprocessing Product Review Text 
# MAGIC - Generate Summary Statistics

# COMMAND ----------

# MAGIC %pip install ydata-profiling==4.0.0

# COMMAND ----------

# python import
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
product_review_raw_df = pd.read_csv("/Volumes/main/default/raw/kaggle_amazon_product_reviews.csv")
profile = ProfileReport(product_review_raw_df, title="Amazon Product Review Data Profile")
profile.to_notebook_iframe()

# COMMAND ----------

# Isolate columns of interest
product_review_raw_df = product_review_raw_df[['id', 'categories', 
                                               'dateAdded', 'dateUpdated',
                                               'dimension', 'weight', 
                                               'manufacturer', 'name', 'prices',
                                               'reviews.date', 'reviews.numHelpful',
                                               'reviews.rating', 'reviews.title',
                                               'reviews.text', 'reviews.username']]

# COMMAND ----------


