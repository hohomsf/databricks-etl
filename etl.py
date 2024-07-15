# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC This is a demonstration of ETL process.<br><br>
# MAGIC This is an interesting dataset about immunization coverage among schools Nova Scotia. The dataset is downloaded here: https://www.kaggle.com/datasets/imtkaggleteam/school-based-immunization-coverage-in-nova-scotia

# COMMAND ----------

# MAGIC %md
# MAGIC # Imort Libraries

# COMMAND ----------

from pyspark.sql import functions as F
import re

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data from DBFS

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/School_Based_Immunization_Coverage_in_Nova_Scotia.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Read CSV into DataFrame
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Investigation

# COMMAND ----------

# MAGIC %md
# MAGIC Before the transformation, I will investigate the dataset to gain an understanding of its quality and determine the necessary transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Schema

# COMMAND ----------

# Schema of the dataset

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Row count

# COMMAND ----------

# Number of rows in the dataset

df.count()

# COMMAND ----------

# Explore the number of Years, Zones and Vaccines
df.groupby("Year").count().sort("Year").display()
df.groupby("Zone").count().sort("Zone").display()
df.groupby("Vaccine").count().sort("Vaccine").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing vaccines?

# COMMAND ----------

# See if each Zone and Vaccine is present in every Year

df.groupby("Year").agg(F.countDistinct("Zone").alias("# Zone"), 
                       F.countDistinct("Vaccine").alias("# Vaccine"))\
                    .sort("Year").display()

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC After viewing the above missing vaccines in each year, I guess the reason may be because of inconsistent vaccine names. For example, HBV vaccine may be labelled as "HBV" in some years but divided into "HBV - Dose 1" and "HBV - Dose 2" in others. This may affect the quality of data analysis.
# MAGIC
# MAGIC I will try to handle this issue later.

# COMMAND ----------

# # Create a DataFrame to store all possible Year and Vaccine combinations
# df_years = df.select(F.col("Year")).distinct()
# df_vaccines = df.select("Vaccine").distinct()

# df_all_vaccines = df_years.join(df_vaccines)

# COMMAND ----------

# # See which Vaccines are missing in each Year
# df_distinct_vaccines = df.select("Year", "Vaccine").distinct()
# df_all_vaccines.exceptAll(df_distinct_vaccines).sort("Year", "Vaccine").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique values check

# COMMAND ----------

# Ensure each unique combination of "Year", "Zone", and "Vaccine" is represented by only one row in the dataset 

df.groupby("Year", "Zone", "Vaccine").count().sort(F.col("count").desc()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rename columns

# COMMAND ----------

# MAGIC %md
# MAGIC In this part, I will convert the column names into snake cases for better readability. Special characters "#" and "%" will also be replaced by "no" and "pct" respectively.

# COMMAND ----------

def to_snake_case(col_name):
    col_name = re.sub(r'(\w)([%#])', r'\1 \2', col_name)

    def replace_special_chars(matchobj):
        char = matchobj.group(0)
        if char == '#':
            return 'no'
        elif char == '%':
            return 'pct'
        elif char.isspace():
            return '_'
        return char

    return re.sub(r'[#% ]', replace_special_chars, col_name).lower()

# COMMAND ----------

rename_cols = [to_snake_case(c) for c in df.columns]

# COMMAND ----------

for old_col, new_col in zip(df.columns, rename_cols):
    df = df.withColumnRenamed(old_col, new_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to proper number format
# MAGIC (e.g. 1,234 to 1234)

# COMMAND ----------

# change "# Immunized" and "# Eligible" columns into int type

num_cols = ["no_immunized", "no_eligible"]

for c in num_cols:
    df = df.withColumn(c, F.regexp_replace(c, ",", "").cast("int"))

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to consistent number format

# COMMAND ----------

# transform "pct_overage" column so the format is consistent with "95_pct_ci" column

df = df.withColumn("pct_coverage", (F.col("pct_coverage") * 100).cast("decimal (4, 1)"))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Separate columns

# COMMAND ----------

# MAGIC %md
# MAGIC "95% CI" is a string column containing a range. However, it is hard to extract useful information from this kind of representation so I am going to separate it into 2 columns with decimal type.

# COMMAND ----------

df = df.withColumn("lower_95_pct_ci", F.split("95_pct_ci", "-").getItem(0).cast("decimal(4, 1)"))\
        .withColumn("upper_95_pct_ci", F.split("95_pct_ci", "-").getItem(1).cast("decimal(4, 1)"))
df = df.drop("95_pct_ci")

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Group vaccines to avoid poor results caused by inconsistent naming

# COMMAND ----------

# MAGIC %md
# MAGIC As mentioned above (please refer to cells # 13 - 16), I believe the inconsistent vaccine names will pose a potential obstacle to our data analysis, so I want to derive a "vaccine_group" column from the values of "vaccine" column.

# COMMAND ----------

# Vaccines can mostly grouped by the string in front of " - "
df = df.withColumn("vaccine_group", F.split(F.col("vaccine"), " - ").getItem(0))

# "MEN-C" is a specific case the cannot be grouped by the above approach
# We need a special adjustment for it w
df = df.withColumn("vaccine_group", F.when(F.col("vaccine_group").like("MEN-C%"), "MEN-C").otherwise(F.col("vaccine_group")))

df.display()

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Using vaccine_group instead of vaccine helps make sure that comparisons among vaccines are meaningful as every group is present in every year. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Load

# COMMAND ----------

# Load the transformed table into the database as delta format
table_name = "ns_school_immunization"

df.write.mode("overwrite").format("delta").saveAsTable(table_name)

# COMMAND ----------

# Try to read the table from the database to ensure it was successfully loaded
spark.read.format('delta').table(table_name).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # More to Explore

# COMMAND ----------

# MAGIC %md
# MAGIC While we have finished the ETL process, I would like to point out the potential problems of this dataset. As shown in the following diagram, we see sudden drops of 50% in HBV and HPV immunization from 2016-17 to 2017-18. However the % coverage does not see significant change until 2019-20.
# MAGIC
# MAGIC We should re-evaluate the quality of this dataset and try to figure out the reasons for that and this will require much more information from the data provider.

# COMMAND ----------

df.display()

# COMMAND ----------

df.display()

# COMMAND ----------


