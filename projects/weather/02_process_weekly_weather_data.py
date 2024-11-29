# Databricks notebook source
# MAGIC %md
# MAGIC # Process Weekly Weather Data
# MAGIC - Ingest raw weather data from NOAA and perform preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC # Precipitation Data
# MAGIC
# MAGIC ## Flag Description
# MAGIC
# MAGIC ### Measurement Flags
# MAGIC - B: precipitation total formed from two 12 hour totals
# MAGIC - D: precipitation total formed from four six hour totals
# MAGIC - H: represents highest or lowest hourly temperature
# MAGIC - K: converted from knots
# MAGIC - L: temperature appears to be lagged with respect to reported hour of observation
# MAGIC - O: converted from oktas
# MAGIC - P: identified as missing presumed zero in DSI 3200 and 3206
# MAGIC - T: trace of precipitation snowfall or snow depth
# MAGIC - W: converted from 16 point WBAN code for wind direction
# MAGIC
# MAGIC
# MAGIC ### Quality Control Flags
# MAGIC - D: failed to duplicate check
# MAGIC - G: failed gap check
# MAGIC - I: failed internal consistency check
# MAGIC - K: failed streak frequent value check
# MAGIC - L: failed check on length of multiday period
# MAGIC - M: failed megaconsistency check
# MAGIC - N: failed naught check
# MAGIC - O: failed climatological outlier check
# MAGIC - R: failed lagged range check
# MAGIC - S: failed spatial consistency check
# MAGIC - T: failed temporal consistency check
# MAGIC - W: failed bounds check
# MAGIC - X: failed climatological outlier check
# MAGIC - Z : flagged as a result of an official Datzilla investigation
# MAGIC
# MAGIC
# MAGIC ### Source Flag
# MAGIC - 0: US Cooperative Summary of the Day
# MAGIC - 6: CDMP Cooperative Summary of the Day
# MAGIC - 7: US Cooperative Summary of the Day Transmitted via WxCoder3
# MAGIC - A: Automated Surface Observing System real time data since January 1 2006
# MAGIC - a: Australian data from the Australian Bureau of Meteorology
# MAGIC - B: US ASOS data for October 2000 to December 2005
# MAGIC - b: Belarus update
# MAGIC - C: Environment Canada
# MAGIC - E: European Climate Assessment and Dataset
# MAGIC - F: US Fort data
# MAGIC - G: Official Global Climate Observing System
# MAGIC - H: High Plains Regional Climate Center real time data
# MAGIC - I: International collection non U.S. data received through personal contacts
# MAGIC - K: US Cooperative Summary of the Day data digitized from paper observer forms
# MAGIC - M: Monthly METAR Extract
# MAGIC - N: Community Collaborative Rain Hail and Snow
# MAGIC - Q: Data from several African countries that had been quarantined
# MAGIC - R: Climate Reference Network and Historical Climatology Network Modernized
# MAGIC - r: All Russian Research Institute of Hydrometeorological Information World Data Center
# MAGIC - S: Global Summary of the Day
# MAGIC - s: China Meteorological Administration National Meteorological Information Center
# MAGIC - T: SNOwpack TELemtry data obtained from the Western Regional Climate Center
# MAGIC - U: Remote Automatic Weather Station data obtained from the Western Regional Climate Center
# MAGIC - u: Ukraine update
# MAGIC - W: WBAN ASOS Summary of the Day from Integrated Surface Data ISD
# MAGIC - X: US First Order Summary of the Day
# MAGIC - Z: Datzilla official additions or replacements
# MAGIC - z: Uzbekistan update

# COMMAND ----------

# import libraries
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window as W
from pyspark.sql import DataFrame, Row
from functools import reduce

import plotly.express as px
import plotly.graph_objs as go

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC - Load NOAA weather data and metadata from ADLS
# MAGIC - Filter out only US based weather measurements

# COMMAND ----------

dev_ex_location = spark.sql("DESCRIBE EXTERNAL LOCATION dev_adls_uc")\
                .filter("name = 'dev_adls_uc'")\
                .select("url").head()[0]

# COMMAND ----------

# load station metadata
## TODO: repath to data-repository in production (02 file location)
schema = T.StructType([T.StructField('station_id', T.StringType()),
                       T.StructField('latitude', T.StringType()),
                       T.StructField('longitude', T.FloatType()),
                       T.StructField('elevation', T.FloatType()),
                       T.StructField('state', T.StringType()),
                       T.StructField('extra_info', T.StringType())])

station_meta = spark.read.format('csv')\
                    .option('header', True)\
                    .schema(schema)\
                    .load(f'{dev_ex_location}/raw/weather/weather_station_meta.csv')
                    
station_meta.display()


# COMMAND ----------

# declare weather schema
weather_schame = T.StructType([ 
                    T.StructField('station_id', T.StringType()),
                    T.StructField('date_key', T.StringType()),
                    T.StructField('measurement_type', T.StringType()),
                    T.StructField('measurement_value', T.FloatType()),
                    T.StructField('measurement_flag', T.StringType()),
                    T.StructField('qc_flag', T.StringType()),
                    T.StructField('source_flag', T.StringType()),
                    T.StructField('hour', T.StringType())
                ])

# COMMAND ----------

# get distinct weather measurement types
weather = spark.read.format('csv')\
                .option('header', True)\
                .schema(weather_schame)\
                .load(f'{dev_ex_location}/raw/weather/precipitation/')\
                .select(F.col("measurement_type")).distinct()
weather.display()

# COMMAND ----------

# load precipitation data
precipitation = spark.read.format('csv')\
                .option('header', True)\
                .schema(weather_schame)\
                .load(f'{dev_ex_location}/raw/weather/precipitation/')\
                .filter("measurement_type IN ('PRCP')").cache()

us_precipitation = precipitation.join(station_meta.select('station_id'), ['station_id'])
us_precipitation.display()

# COMMAND ----------

# load precipitation data
temperature = spark.read.format('csv')\
                .option('header', True)\
                .schema(weather_schame)\
                .load(f'{dev_ex_location}/raw/weather/precipitation/')\
                .filter("measurement_type IN ('TAVG', 'TMIN', 'TMAX')").cache()

us_temperature = temperature.join(station_meta.select('station_id'), ['station_id'])
us_temperature.display()

# COMMAND ----------

# load precipitation data
windspeed = spark.read.format('csv')\
                .option('header', True)\
                .schema(weather_schame)\
                .load(f'{dev_ex_location}/raw/weather/precipitation/')\
                .filter("measurement_type IN ('AWND')").cache()

us_windspeed = windspeed.join(station_meta.select('station_id'), ['station_id'])
us_windspeed.display()

# COMMAND ----------

# load precipitation data
total_sunsine_minutes = spark.read.format('csv')\
                .option('header', True)\
                .schema(weather_schame)\
                .load(f'{dev_ex_location}/raw/weather/precipitation/')\
                .filter("measurement_type IN ('TSUN')").cache()

us_total_sunsine_minutes = total_sunsine_minutes.join(station_meta.select('station_id'), ['station_id'])
us_total_sunsine_minutes.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing
# MAGIC - Pivot Table
# MAGIC - Clean up some weird values
# MAGIC - Drop some NULL measurements

# COMMAND ----------

# pivot the measurement to obtain daily level precipitation 
# create datetime column
# drop any remaining null measurements
us_station_precipitation_pivot = us_precipitation.groupby('station_id','date_key')\
                                .pivot('measurement_type')\
                                .agg(F.max('measurement_value').alias('measurement_value'),
                                F.max('measurement_flag').alias('measurement_flag'),
                                F.max('qc_flag').alias('qc_flag'),
                                F.max('source_flag').alias('source_flag'))\
                                .withColumn('date_key', F.trim(F.col('date_key'))) \
                                .withColumn('year', F.expr("substr(date_key, 0, 4)")) \
                                .withColumn('date', F.to_date(F.concat(F.expr("substr(date_key, 0, 4)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 5, 2)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 7, 2)")))) \
                                .drop(*["null_measurement_value", 
                                        "null_measurement_flag", 
                                        "null_qc_flag", 
                                        "null_source_flag"])\
                                .filter(F.col('date_key').isNotNull()).cache()
us_station_precipitation_pivot.display() 

# COMMAND ----------

# pivot the measurement to obtain daily level temperature for avg, min, and max 
# temperature are in degree Celcius, but NOAA scale it by 10x for some reason???
us_station_temperature_pivot = us_temperature.groupby('station_id','date_key')\
                                .pivot('measurement_type')\
                                .agg(F.max('measurement_value').alias('measurement_value'),
                                    F.max('measurement_flag').alias('measurement_flag'),
                                    F.max('qc_flag').alias('qc_flag'),
                                    F.max('source_flag').alias('source_flag'))\
                                .withColumn('TMAX_measurement_value', F.col("TMAX_measurement_value").cast('int')/10) \
                                .withColumn('TMIN_measurement_value', F.col("TMIN_measurement_value").cast('int')/10) \
                                .withColumn('TAVG_measurement_value', F.col("TAVG_measurement_value").cast('int')/10) \
                                .withColumn('date_key', F.trim(F.col('date_key'))) \
                                .withColumn('year', F.expr("substr(date_key, 0, 4)")) \
                                .withColumn('date', F.to_date(F.concat(F.expr("substr(date_key, 0, 4)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 5, 2)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 7, 2)")))) \
                                .drop(*["null_measurement_value", 
                                        "null_measurement_flag", 
                                        "null_qc_flag", 
                                        "null_source_flag"])\
                                .filter(F.col('date_key').isNotNull()).cache()
us_station_temperature_pivot.display()

# COMMAND ----------

# pivot the measurement to obtain daily level precipitation 
# create datetime column
# drop any remaining null measurements
us_station_windspeed_pivot = us_windspeed.groupby('station_id','date_key')\
                                .pivot('measurement_type')\
                                .agg(F.max('measurement_value').alias('measurement_value'),
                                    F.max('measurement_flag').alias('measurement_flag'),
                                    F.max('qc_flag').alias('qc_flag'),
                                    F.max('source_flag').alias('source_flag'))\
                                .withColumn('date_key', F.trim(F.col('date_key'))) \
                                .withColumn('year', F.expr("substr(date_key, 0, 4)")) \
                                .withColumn('date', F.to_date(F.concat(F.expr("substr(date_key, 0, 4)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 5, 2)"), F.lit('-'), 
                                                                        F.expr("substr(date_key, 7, 2)")))) \
                                .drop(*["null_measurement_value", 
                                        "null_measurement_flag", 
                                        "null_qc_flag", 
                                        "null_source_flag"])\
                                .filter(F.col('date_key').isNotNull()).cache()
us_station_windspeed_pivot.display() 

# COMMAND ----------

# pivot the measurement to obtain daily level precipitation 
# create datetime column
# drop any remaining null measurements
us_station_total_sunsine_minutes_pivot = us_total_sunsine_minutes.groupby('station_id','date_key')\
                                              .pivot('measurement_type')\
                                              .agg(F.max('measurement_value').alias('measurement_value'),
                                                  F.max('measurement_flag').alias('measurement_flag'),
                                                  F.max('qc_flag').alias('qc_flag'),
                                                  F.max('source_flag').alias('source_flag'))\
                                              .withColumn('date_key', F.trim(F.col('date_key'))) \
                                              .withColumn('year', F.expr("substr(date_key, 0, 4)")) \
                                              .withColumn('date', F.to_date(F.concat(F.expr("substr(date_key, 0, 4)"), F.lit('-'), 
                                                                                      F.expr("substr(date_key, 5, 2)"), F.lit('-'), 
                                                                                      F.expr("substr(date_key, 7, 2)")))) \
                                              .drop(*["null_measurement_value", 
                                                      "null_measurement_flag", 
                                                      "null_qc_flag", 
                                                      "null_source_flag"])\
                                              .filter(F.col('date_key').isNotNull()).cache()
us_station_total_sunsine_minutes_pivot.display() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to ADLS

# COMMAND ----------

us_station_precipitation_pivot.repartition(512).write.mode('overwrite')\
                              .parquet(f'{dev_ex_location}/processed/weather/precipitation')

# COMMAND ----------

us_station_temperature_pivot.repartition(512).write.mode('overwrite')\
                              .parquet(f'{dev_ex_location}/processed/weather/temperature')

# COMMAND ----------

us_station_windspeed_pivot.repartition(512).write.mode('overwrite')\
                              .parquet(f'{dev_ex_location}/processed/weather/windspeed')

# COMMAND ----------

us_station_total_sunsine_minutes_pivot.repartition(512).write.mode('overwrite')\
                              .parquet(f'{dev_ex_location}/processed/weather/total_sunsine_minutes')

# COMMAND ----------


