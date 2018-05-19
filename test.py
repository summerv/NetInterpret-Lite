import os

os.environ["SPARK_HOME"] = "/home/vicky/spark-2.3.0-bin-hadoop2.7/"
os.environ["PYSPARK_PYTHON"] = "/home/vicky/anaconda3/bin/python"

from pyspark import SparkContext, HiveContext
from pyspark.sql import Row, SparkSession
import numpy as np
from pyspark.sql.types import *
import time
import random
import pyspark.sql.functions

sc = SparkContext()
spark = SparkSession(sc) \
    .builder \
    .master("local") \
    .appName("NetInterpret-Storage") \
    .enableHiveSupport() \
    .getOrCreate()
MATRIX_SCHEMA = StructType([
    StructField("id_1", LongType(), True),
    StructField("id_2", LongType(), True),
    StructField("id_3", LongType(), True),
    StructField("id_4", LongType(), True),
    StructField("value", DoubleType(), True)])

arr = np.array(range(120)).reshape(2, 3, 4, 5)
first_dim = sc.parallelize(range(2))
mat = first_dim.flatMap(lambda x: [(x, i, j, k, float(arr[x][i][j][k])) for i in range(3) for j in range(4) for k in range(5)])
df = spark.createDataFrame(mat, MATRIX_SCHEMA)
# df.createOrReplaceTempView("temp")
# res = spark.sql("create table feature_map as select * from temp")

# df.write.option("path", "hdfs://user/hive/warehouse/feature-map-test").format("text").saveAsTable("feature-map-test")
# df.createOrReplaceTempView("my_temp_table")
# hc = HiveContext(sc)
# hc.sql("drop table if exists my_table")
# spark.sql("create table feature_map as select * from my_temp_table")
df.write.format('parquet').mode("overwrite").saveAsTable("feature_map2")
res = spark.sql("select * from feature_map2")
res.show(120)
print(res.count())