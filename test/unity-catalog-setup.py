# Databricks notebook source
# MAGIC %md
# MAGIC # External Locations

# COMMAND ----------

# MAGIC %md
# MAGIC ## DEV ADLS
# MAGIC - Read/Write Permission

# COMMAND ----------

dev_ex_location = spark.sql("DESCRIBE EXTERNAL LOCATION dev_adls_uc")\
                .filter("name = 'dev_adls_uc'")\
                .select("url").head()[0]
dev_ex_location

# COMMAND ----------

# test write to external location
dbutils.fs.put(dev_ex_location + "test.txt", "", True)

# COMMAND ----------

# test read from External Location
dbutils.fs.ls(dev_ex_location)

# COMMAND ----------

# test load data from external location
df = spark.read.format("csv")\
                .option("header", True)\
                .option("inferSchema", True)\
                .load(dev_ex_location+"/raw/kaggle_amazon_product_reviews.csv")
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PROD ADLS
# MAGIC - Read Permission

# COMMAND ----------

prod_ex_location = spark.sql("DESCRIBE EXTERNAL LOCATION prod_adls_uc")\
                .filter("name = 'prod_adls_uc'")\
                .select("url").head()[0]
prod_ex_location

# COMMAND ----------

# test write to a read only external location
dbutils.fs.put(prod_ex_location + "test.txt", "", True)

# COMMAND ----------

# test read from External Location
dbutils.fs.ls(prod_ex_location)

# COMMAND ----------

# test load data from a read only external location
df = spark.read.format("csv")\
                .option("header", True)\
                .option("inferSchema", True)\
                .load(prod_ex_location+"/raw/kaggle_amazon_product_reviews.csv")
df.display()

# COMMAND ----------



# COMMAND ----------

data = [
    {"Category": "A", "ID": 1, "Value": 121.44, "Truth": True},
    {"Category": "B", "ID": 2, "Value": 300.01, "Truth": False},
    {"Category": "C", "ID": 3, "Value": 10.99, "Truth": None},
    {"Category": "E", "ID": 4, "Value": 33.87, "Truth": True},
]
df = spark.createDataFrame(data)
df.write.mode("overwrite").saveAsTable(f"main.dev_uc.test_output")

# COMMAND ----------


