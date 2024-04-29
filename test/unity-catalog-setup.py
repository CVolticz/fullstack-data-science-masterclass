# Databricks notebook source
# MAGIC %md
# MAGIC # External Locations

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTERNAL LOCATION metastore_root_location

# COMMAND ----------

location = spark.sql("DESCRIBE EXTERNAL LOCATION metastore_root_location")\
                .filter("name = 'metastore_root_location'")\
                .select("url").head()[0]
location

# COMMAND ----------

ex_location = "abfss://outbound@p1databricksadls.dfs.core.windows.net/"

# COMMAND ----------

# test write to external location
dbutils.fs.put(ex_location + "test.txt", "", True)

# COMMAND ----------

# test read from External Location
dbutils.fs.ls(ex_location)

# COMMAND ----------

# test load data from external location
test_df = spark.read.format("csv")\
                .option("header", True)\
                .option("inferSchema", True)\
                .load(ex_location+"test_data.csv")
test_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Volumes

# COMMAND ----------

# test write to volumn
dbutils.fs.put("/Volumes/main/default/raw/test.txt", "", True)

# COMMAND ----------


