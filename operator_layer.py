import os

os.environ["SPARK_HOME"] = "/home/vicky/spark-2.3.0-bin-hadoop2.7/"
os.environ["PYSPARK_PYTHON"] = "/home/vicky/anaconda3/bin/python"

from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
import numpy as np
from pyspark.sql.types import *
import time
import random
import pyspark.sql.functions
from storage_layer import StorageLayer
import settings


class OperatorLayer:
    def __init__(self, size):
        self.sl = StorageLayer()
        self.feature_size = size
        self.wholefeatures = self.load_features()    # dataframe


    def load_features(self):
        load_start_time = time.time()
        feature_rdd = self.sl.spark.read \
                                    .format("com.databricks.spark.csv") \
                                    .option("header", "false") \
                                    .option("inferSchema", "true") \
                                    .load("hdfs://localhost:9000/user/data/dataset/test/layer4_500.csv").rdd

        df = self.sl.spark.createDataFrame(feature_rdd, self.sl.MATRIX_SCHEMA)
        print("load features on size %d using time: %f s." % (self.feature_size, time.time() - load_start_time))
        return df

    def load_features2(self):
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")
        mmap_files = [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name) for feature_name in
                      settings.FEATURE_NAMES]

        skip = True

        if os.path.exists(features_size_file):
            features_size = np.load(features_size_file)
        else:
            skip = False

        for i, mmap_file in enumerate(mmap_files):
            if os.path.exists(mmap_file) and features_size is not None:
                print('loading features %s' % settings.FEATURE_NAMES[i])
                wholefeatures[i] = np.memmap(mmap_file, dtype=float, mode='r', shape=tuple(features_size[i]))
            else:
                print('ERROR: %s file missing!' % settings.FEATURE_NAMES[i])
                skip = False

        return wholefeatures

    def get_quantile_threshold(self, layer_i=0, quantile_percent=0.95):
        # wholefeatures_2d = np.memmap.transpose(self.wholefeatures[layer_i], [0, 2, 3, 1])
        # wholefeatures_2d = wholefeatures_2d.reshape((-1, 512), order='C')

        # print("start local to dataframe...")
        # wholefeatures_df = self.sl.local_to_dataframe(self.wholefeatures[layer_i][:test_size])
        # print("end local to dataframe...")

        self.wholefeatures.createOrReplaceTempView("select_res")
        #select_query = "select * from wholefeatures_df where id_1 < " + str(self.feature_size)
        #select_res = self.sl.spark.sql(select_query)
        #select_res.createOrReplaceTempView("select_res")

        quantile_query = "select id_4, percentile_approx(value, " + str(quantile_percent) + ") as appoxQuantile from select_res group by id_4"

        # print("start spark.sql ...")
        quantile_threshold_start_time = time.time()
        quantile_res = self.sl.spark.sql(quantile_query)
        print("quantile threshold on size %d using time: %f s." % (self.feature_size, time.time() - quantile_threshold_start_time))
        return quantile_res


if __name__ == '__main__':
    ol = OperatorLayer(500)
    res = ol.get_quantile_threshold()
