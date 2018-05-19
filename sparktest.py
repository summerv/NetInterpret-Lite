"""SimpleApp"""
import os

os.environ["SPARK_HOME"] = "/home/vicky/spark-2.3.0-bin-hadoop2.7/"
os.environ["PYSPARK_PYTHON"] = "/home/vicky/anaconda3/bin/python"


from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext, Row, SparkSession
# from scipy.misc import imread
# from pyspark import SparkFiles
from pyspark.ml.image import ImageSchema
import numpy as np
from pyspark.mllib.linalg import DenseMatrix
from pyspark.sql.types import *

warehouseLocation = "hdfs://user/hive/warehouse"
# warehouseLocation = "file:k/user/hive/warehouse"
sc = SparkContext()
spark = SparkSession(sc).builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.sql.warehouse.dir", warehouseLocation) \
        .enableHiveSupport() \
        .getOrCreate()
print(spark.catalog.listTables())
# result = spark.sql("select * from broden_index")    # returns a DataFrame
# rc = result.collect()
# result.show(10)
# print(result.dtypes)

MATRIX_SCHEMA = StructType([
    StructField("id_1", LongType(), True),
    StructField("id_2", LongType(), True),
    StructField("id_3", LongType(), True),
    StructField("id_4", LongType(), True),
    StructField("value", DoubleType(), True)])

arr = np.array(range(120), dtype=float).reshape(2, 3, 4, 5)
dim_1 = 2
dim_2 = 3
dim_3 = 4
dim_4 = 5
data = [Row(id_1=a, id_2=b, id_3=c, id_4=d, value=float(arr[a][b][c][d]))
        for a in range(dim_1) for b in range(dim_2)
        for c in range(dim_3) for d in range(dim_4)]
df = spark.createDataFrame(data, MATRIX_SCHEMA)
n = df.groupBy().max('id_2').collect()[0]['max(id_2)']
mat_reverse = df.select(df.id_1, (n-df.id_2).alias("id_2"), df.id_3, df.id_4, df.value)
mat_reverse.show(120)

maxvalue = df.groupBy([df.id_1, df.id_3, df.id_4]).max('value').collect()  #[0]['max(value)']
print('maxvalue: ', maxvalue)

# first_dim = sc.parallelize(range(dim_1))
# mat = first_dim.flatMap(lambda x : [(x, i, j, k, 0.0) for i in range(dim_2) for j in range(dim_3) for k in range(dim_4)])
# df2 = spark.createDataFrame(mat, MATRIX_SCHEMA)
# df.show(20)
# print(df.count())

################## dataframe design 4.2.5 ~ 4.2.7 #################
FEATUREMAP_SCHEMA = StructType([
    StructField("model_name", StringType(), True),
    StructField("layer_name", StringType(), True),
    StructField("feature_map", ArrayType(ArrayType(ArrayType(ArrayType(DoubleType())))), True)])

THRESHOLDS_SCHEMA = StructType([
    StructField("dataset", StringType(), True),
    StructField("model_name", StringType(), True),
    StructField("layer_name", StringType(), True),
    StructField("unit_id", IntegerType(), True),
    StructField("thresh", DoubleType(), True)])

TALLY_RESULT_SCHEMA = StructType([
    StructField("dataset", StringType(), True),
    StructField("model_name", StringType(), True),
    StructField("layer_name", StringType(), True),
    StructField("unit_id", IntegerType(), True),
    StructField("semantic_label", StringType(), True)])

tmp = [('resnet18_places365', 'layer4', [[[[1.0, 2.0], [2.0, 3.0]], [[-1.0, -2.0], [-2.0, 8.0]]],
                                        [[[1.0, 2.0], [2.0, 3.0]], [[-1.0, -2.0], [-2.0, 8.0]]]])]
tmp_df = spark.createDataFrame(tmp, FEATUREMAP_SCHEMA)
result = tmp_df.collect()
# print(result)

############### image read test ######################

df = ImageSchema.readImages('/user/data/dataset/val_256/images_224', recursive=False)
# print(df.count())
# print(ImageSchema.imageSchema)
df.show()
print(df.describe())
df_ = df.collect()
print("hi")
print(df.count())
# print(df)
df0 = df_[0]
# print(df0)
imgnd = ImageSchema.toNDArray(df0.image)
print(imgnd)
print(imgnd.shape)


###################### matrix test ######################
m = DenseMatrix(2, 2, range(4))
mnd = m.toArray()
# print(mnd)