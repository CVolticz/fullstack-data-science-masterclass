# Databricks notebook source
# MAGIC %md
# MAGIC # Precipitation Data Ingestion
# MAGIC - Read in precipitation data from NOAA weather and store it in ADLS

# COMMAND ----------

import os
import re
import pandas as pd
import numpy as np
from ftplib import FTP
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_data_year(year, output_dir):
    """
        Function to Extract NOAA Data by Year
        Inputs:
            - year {string}: year of data extraction
            - ouput_dir {string}: path to store output files
    """
    print("\nGETTING DATA FOR YEAR: ",year)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the filename and local path
    filename = f"{year}.csv.gz"
    local_path = os.path.join(output_dir, filename)

    # Call onto NOAA FTP Server
    # Get public GHCN Daily data by year
    # Then save to specific directory
    try:
        ftp = FTP('ftp.ncdc.noaa.gov')
        ftp.login()
        ftp.cwd('pub/data/ghcn/daily/by_year')

        # Download the file
        with open(local_path, 'wb') as file:
            ftp.retrbinary(f'RETR {filename}', file.write)
        print(f"Downloaded {filename} to {output_dir}")

    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        
    finally:
        # Close the FTP connection
        ftp.quit()

# COMMAND ----------

def get_ghcnd_stations(area='US'):
    """
        Function to grab GHCN station metadata
        Inputs:
            - area {string}: global area to filter on
        Outputs:
            - stations_metadata {pd.DataFrame}: dataframe containing station metadata
    """
    print("\nGRABBING LATEST STATION METADATA FILE")


    # Call onto NOAA FTP Server
    # Get public GHCN metadata by year
    ftp = FTP('ftp.ncdc.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/ghcn/daily')
    ftp.retrbinary('RETR ghcnd-stations.txt', open(f'/tmp/ghcnd-stations.txt', 'wb').write)
    ftp.quit()

    # Read in GHCND-D Stations File
    # Format it properly
    # filter out specific area region
    ghcnd_stnfile=f'/tmp/ghcnd-stations.txt'
    ghcnd_stations= np.genfromtxt(ghcnd_stnfile,delimiter=(11,9,10,7,4,30),dtype=str)
    filtered_stations = [s for s in ghcnd_stations if re.search(area, s[0])]

    # organize the station data as a dataframe
    stations_metadata = pd.DataFrame(data = filtered_stations,
                                    columns=['station_id','latitude','longitude',
                                             'elevation','state','extra_info'])

    return stations_metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Extraction
# MAGIC - initialize parameters
# MAGIC - ingest data from NOAA API
# MAGIC - store data in ADLS

# COMMAND ----------

ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
ftp.cwd('pub/data/ghcn/daily/by_year')

# COMMAND ----------

output_dir = f"/Volumes/main/dev_uc/raw/weather"

# COMMAND ----------

# Get weather station data and save to adls
station_meta = get_ghcnd_stations()

# save to adls
df, fname = station_meta, '/weather_station_meta.csv'
outbound_fname = output_dir + fname
df.to_csv(outbound_fname, header = True, index = False)

# COMMAND ----------

# # get precipitation data and save it to adls
current_year = datetime.now().year
get_data_year(year=current_year, output_dir=output_dir+'/NOAA')


# COMMAND ----------

dbutils.notebook.exit("Success")

# COMMAND ----------

# Load other years worth of data
for year in ['2016']:
    get_data_year(year=year, output_dir=output_dir+'/NOAA')

# COMMAND ----------


