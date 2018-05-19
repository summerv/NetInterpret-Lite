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


class StorageLayer:
    def __init__(self):
        # self.warehouseLocation = "hdfs://user/hive/warehouse"
        self.sc = SparkContext()
        self.spark = SparkSession(self.sc) \
            .builder \
            .master("local") \
            .appName("NetInterpret-Storage") \
            .enableHiveSupport() \
            .getOrCreate()
        self.MATRIX_SCHEMA = StructType([
            StructField("id_1", LongType(), True),
            StructField("id_2", LongType(), True),
            StructField("id_3", LongType(), True),
            StructField("id_4", LongType(), True),
            StructField("value", DoubleType(), True)])

    def zeroes(self, dim_1, dim_2, dim_3, dim_4):
        # zero_mat = [Row(id_1=a, id_2=b, id_3=c, id_4=d, value=0.0)
        #             for a in range(dim_1)
        #             for b in range(dim_2)
        #             for c in range(dim_3)
        #             for d in range(dim_4)]
        # return self.spark.createDataFrame(zero_mat, self.MATRIX_SCHEMA)

        first_dim = self.sc.parallelize(range(dim_1))
        mat = first_dim.flatMap(lambda x: [(x, i, j, k, 0.0) for i in range(dim_2) for j in range(dim_3) for k in range(dim_4)])
        return self.spark.createDataFrame(mat, self.MATRIX_SCHEMA)

    def ones_2d(self, dim_3, dim_4):
        first_dim = self.sc.parallelize([-1])
        mat = first_dim.flatMap(lambda x: [(x, -1, j, k, 1.0) for j in range(dim_3) for k in range(dim_4)])
        return self.spark.createDataFrame(mat, self.MATRIX_SCHEMA)

    def random_2d(self, dim_3, dim_4, random_start=1, random_end=5):
        first_dim = self.sc.parallelize([-1])
        mat = first_dim.flatMap(lambda x: [(x, -1, j, k, float(random.randint(random_start, random_end))) for j in range(dim_3) for k in range(dim_4)])
        return self.spark.createDataFrame(mat, self.MATRIX_SCHEMA)

    def reverse(self, mat, idx):
        '''
        :param mat:
        :param idx: idx = 0, 1, 2 or 3
        :return:
        '''
        mat_shape = self.shape(mat)
        if idx == 0:
            return mat.select((mat_shape[0] - 1 - mat.id_1).alias("id_1"), mat.id_2, mat.id_3, mat.id_4, mat.value)
        elif idx == 1:
            return mat.select(mat.id_1, (mat_shape[1] - 1 - mat.id_2).alias("id_2"), mat.id_3, mat.id_4, mat.value)
        elif idx == 2:
            return mat.select(mat.id_1, mat.id_2, (mat_shape[2] - 1 - mat.id_3).alias("id_3"), mat.id_4, mat.value)
        else:
            return mat.select(mat.id_1, mat.id_2, mat.id_3, (mat_shape[3] - 1 - mat.id_4).alias("id_4"), mat.value)

    def max(self, mat, idx=None):
        '''
        :param self:
        :param mat:
        :param idx: idx = 0, 1, 2 or 3
        :return:
        '''
        if idx == None:
            return mat.groupBy().max('value').collect()[0]['max(value)']
        if idx == 0:
            tmp = mat.groupBy([mat.id_2, mat.id_3, mat.id_4]).max('value')
            return tmp.withColumn('id_1', lit(-1)).withColumnRenamed("max(value)", "value")
        elif idx == 1:
            tmp = mat.groupBy([mat.id_1, mat.id_3, mat.id_4]).max('value')
            return tmp.withColumn('id_2', lit(-1)).withColumnRenamed("max(value)", "value")
        elif idx == 2:
            tmp = mat.groupBy([mat.id_1, mat.id_2, mat.id_4]).max('value')
            return tmp.withColumn('id_3', lit(-1)).withColumnRenamed("max(value)", "value")
        else:
            tmp = mat.groupBy([mat.id_1, mat.id_2, mat.id_3]).max('value')
            return tmp.withColumn('id_4', lit(-1)).withColumnRenamed("max(value)", "value")

    def transpose(self, mat, axes=None):
        '''
        :param mat: dataframe
        :param axes: axis sample: [0,3,1,2]
        :return:
        '''
        column_list = ["id_1", "id_2", "id_3", "id_4"]
        trans_map = {}
        for i, axe in enumerate(axes):
            trans_map[axe] = column_list[i]
        return mat.select(mat.id_1.alias(trans_map[0]),
                          mat.id_2.alias(trans_map[1]),
                          mat.id_3.alias(trans_map[2]),
                          mat.id_4.alias(trans_map[3]),
                          mat.value)

    def add(self, mat1, mat2):
        '''
        :param mat1:
        :param mat2: can be a constant(int/float) or a matrix with same size of mat1
        :return:
        '''
        mat1_shape = self.shape(mat1)
        mat2_shape = self.shape(mat2)
        if mat1_shape != mat2_shape:
            raise Exception("Error: mat1 and mat2 are not the same size!")

        if type(mat2) == int or type(mat2) == float:
            return mat1.select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value + mat2).alias("value"))

        join_condition = [mat1.id_1 == mat2.id_1, mat1.id_2 == mat2.id_2,
                          mat1.id_3 == mat2.id_3, mat1.id_4 == mat2.id_4]

        return mat1.join(mat2, join_condition) \
            .select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value + mat2.value).alias("value")) \
            .orderBy(["id_1", "id_2", "id_3", "id_4"])

    def subtract(self, mat1, mat2):
        '''
        :param mat1:
        :param mat2: can be a constant(int/float) or a matrix with same size of mat1
        :return:
        '''
        mat1_shape = self.shape(mat1)
        mat2_shape = self.shape(mat2)
        if mat1_shape != mat2_shape:
            raise Exception("Error: mat1 and mat2 are not the same size!")

        if type(mat2) == int or type(mat2) == float:
            return mat1.select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value + mat2).alias("value"))

        join_condition = [mat1.id_1 == mat2.id_1, mat1.id_2 == mat2.id_2,
                          mat1.id_3 == mat2.id_3, mat1.id_4 == mat2.id_4]

        return mat1.join(mat2, join_condition) \
            .select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value - mat2.value).alias("value")) \
            .orderBy(["id_1", "id_2", "id_3", "id_4"])

    def dot_multiply(self, mat1, mat2):
        '''
        :param mat1:
        :param mat2: matrix with the same size of mat1
        :return:
        '''
        mat1_shape = self.shape(mat1)
        mat2_shape = self.shape(mat2)
        if mat1_shape != mat2_shape:
            raise Exception("Error: mat1 and mat2 are not the same size!")

        join_condition = [mat1.id_1 == mat2.id_1, mat1.id_2 == mat2.id_2, mat1.id_3 == mat2.id_3,
                          mat1.id_4 == mat2.id_4]
        return mat1.join(mat2, join_condition) \
            .select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value * mat2.value).alias("value")) \
            .orderBy(["id_1", "id_2", "id_3", "id_4"])

    def dot_divide(self, mat1, mat2):
        '''
        :param mat1:
        :param mat2: matrix with the same size of mat1
        :return:
        '''
        mat1_shape = self.shape(mat1)
        mat2_shape = self.shape(mat2)
        if mat1_shape != mat2_shape:
            raise Exception("Error: mat1 and mat2 are not the same size!")

        join_condition = [mat1.id_1 == mat2.id_1, mat1.id_2 == mat2.id_2, mat1.id_3 == mat2.id_3,
                          mat1.id_4 == mat2.id_4]
        return mat1.join(mat2, join_condition) \
            .select(mat1.id_1, mat1.id_2, mat1.id_3, mat1.id_4, (mat1.value / mat2.value).alias("value")) \
            .orderBy(["id_1", "id_2", "id_3", "id_4"])

    def multiply_2d(self, mat1, mat2):
        temp = mat1.crossJoin(mat2).filter(mat1.id_4 == mat2.id_3) \
            .select(mat1.id_3, mat2.id_4, (mat1.value * mat2.value).alias("value"))

        temp.createOrReplaceTempView("temp")
        return self.spark.sql("SELECT -1 AS id_1, -1 AS id_2, id_3, id_4, sum(value) AS value FROM temp GROUP BY id_3, id_4 ORDER BY id_3, id_4")

    def local_to_dataframe(self, mat):
        '''
        :param mat: can be local array, np array or memmap
        :return:
        '''
        mat_shape = mat.shape
        print("mat_shape: ", mat_shape)
        if len(mat_shape) == 1:
            # mat = [Row(id_1=-1, id_2=-1, id_3=-1, id_4=d, value=float(mat[d]))
            #        for d in range(mat_shape[0])]
            first_dim = self.sc.parallelize([-1])
            mat_rdd = first_dim.flatMap(
                lambda x: [(x, -1, -1, k, float(mat[k])) for k in range(mat_shape[-1])])

        if len(mat_shape) == 2:
            # mat = [Row(id_1=-1, id_2=-1, id_3=c, id_4=d, value=float(mat[c][d]))
            #        for c in range(mat_shape[0])
            #        for d in range(mat_shape[1])]
            first_dim = self.sc.parallelize([-1])
            mat_rdd = first_dim.flatMap(
                lambda x: [(x, -1, j, k, float(mat[j][k])) for j in range(mat_shape[-2]) for k in range(mat_shape[-1])])

        if len(mat_shape) == 3:
            # mat = [Row(id_1=-1, id_2=b, id_3=c, id_4=d, value=float(mat[b][c][d]))
            #        for b in range(mat_shape[0])
            #        for c in range(mat_shape[1])
            #        for d in range(mat_shape[2])]
            first_dim = self.sc.parallelize([-1])
            mat_rdd = first_dim.flatMap(
                lambda x: [(x, i, j, k, float(mat[i][j][k])) for i in range(mat_shape[-3]) for j in range(mat_shape[-2]) for k in range(mat_shape[-1])])

        if len(mat_shape) == 4:
            # mat = [Row(id_1=a, id_2=b, id_3=c, id_4=d, value=float(mat[a][b][c][d]))
            #        for a in range(mat_shape[0])
            #        for b in range(mat_shape[1])
            #        for c in range(mat_shape[2])
            #        for d in range(mat_shape[3])]

            first_dim = self.sc.parallelize(range(mat_shape[-4]))
            mat_rdd = first_dim.flatMap(
                lambda x: [(x, i, j, k, float(mat[x][i][j][k])) for i in range(mat_shape[-3]) for j in range(mat_shape[-2]) for k in range(mat_shape[-1])])

        return self.spark.createDataFrame(mat_rdd, self.MATRIX_SCHEMA)


def function_test():
    storage_layer = StorageLayer()
    dim_1 = 1
    dim_2 = 2
    dim_3 = 3
    dim_4 = 4
    arr1 = np.array(range(24), dtype=float).reshape(dim_1, dim_2, dim_3, dim_4)
    arr2 = np.array(range(120, 120 + 24, 1), dtype=float).reshape(dim_1, dim_2, dim_3, dim_4)
    arr3 = np.array(range(20), dtype=float).reshape(4, 5)
    arr4 = np.array(range(30), dtype=float).reshape(5, 6)
    data1 = [Row(id_1=a, id_2=b, id_3=c, id_4=d, value=float(arr1[a][b][c][d]))
             for a in range(dim_1) for b in range(dim_2)
             for c in range(dim_3) for d in range(dim_4)]
    data2 = [Row(id_1=a, id_2=b, id_3=c, id_4=d, value=float(arr2[a][b][c][d]))
             for a in range(dim_1) for b in range(dim_2)
             for c in range(dim_3) for d in range(dim_4)]
    data3 = [Row(id_1=-1, id_2=-1, id_3=c, id_4=d, value=float(arr3[c][d]))
             for c in range(4) for d in range(5)]
    data4 = [Row(id_1=-1, id_2=-1, id_3=c, id_4=d, value=float(arr4[c][d]))
             for c in range(5) for d in range(6)]
    data_df1 = storage_layer.spark.createDataFrame(data1, storage_layer.MATRIX_SCHEMA)
    data_df2 = storage_layer.spark.createDataFrame(data2, storage_layer.MATRIX_SCHEMA)
    data_df3 = storage_layer.spark.createDataFrame(data3, storage_layer.MATRIX_SCHEMA)
    data_df4 = storage_layer.spark.createDataFrame(data4, storage_layer.MATRIX_SCHEMA)

    # test1: zeroes
    res1 = storage_layer.zeroes(1, 2, 3, 4)
    res1.show(dim_1 * dim_2 * dim_3 * dim_4)

    # test2: reverse
    res2_0 = storage_layer.reverse(data_df1, 0)
    res2_1 = storage_layer.reverse(data_df1, 1)
    res2_2 = storage_layer.reverse(data_df1, 2)
    res2_3 = storage_layer.reverse(data_df1, 3)
    # print("test2:")
    res2_0.show(dim_1 * dim_2 * dim_3 * dim_4)
    res2_1.show(dim_1 * dim_2 * dim_3 * dim_4)
    res2_2.show(dim_1 * dim_2 * dim_3 * dim_4)
    res2_3.show(dim_1 * dim_2 * dim_3 * dim_4)

    # test3: max
    # print("test3_0:")
    # the two expression below are equal to each other
    res3_0 = storage_layer.max(data_df1)
    # print(res3_0)
    # print(np.max(arr1))

    # print("test3_1:")
    res3_2 = storage_layer.max(data_df1, 1)
    # print(res3_2.show(24))
    # print(np.max(arr1, 1))

    # print("test3_2:")
    # the two expression below are equal to each other
    res3_3 = storage_layer.max(storage_layer.max(data_df1, 1), 3)  # pay attention, 3!
    res3_3.show()
    # print(np.max(np.max(arr1, 1), 2))

    # test4: transpose
    # print("test4:")
    # the two expression below are equal to each other
    res4 = storage_layer.transpose(data_df1, [0, 3, 1, 2])
    res4.show(24)
    # print(arr1.transpose([0, 3, 1, 2]))

    # test5: add
    # print("test5:")
    # the two expression below are equal to each other
    res5_0 = storage_layer.add(data_df1, data_df2)
    res5_0.show(24)
    # print(arr1 + arr2)
    # raise exception
    # res5_1 = storage_layer.add(data_df1, data_df3)

    # test6: subtract
    # print("test6:")
    # the two expression below are equal to each other
    res6_0 = storage_layer.subtract(data_df1, data_df2)
    res6_0.show(24)
    # print(arr1 - arr2)
    # raise exception
    # res6_1 = storage_layer.subtract(data_df1, data_df3)

    # test7: dot_multiply
    # print("test7:")
    # the two expression below are equal to each other
    res7_0 = storage_layer.dot_multiply(data_df1, data_df2)
    res7_0.show(24)
    # print(arr1 * arr2)
    # raise exception
    # res7_1 = storage_layer.dot_multiply(data_df1, data_df3)

    # test8: dot_divide
    # print("test8:")
    # the two expression below are equal to each other
    res8_0 = storage_layer.dot_divide(data_df1, data_df2)
    res8_0.show(24)
    # print(arr1 / arr2)
    # raise exception
    # res8_1 = storage_layer.dot_divide(data_df1, data_df3)

    # test9: multiply
    # print("test9:")
    res9_0 = storage_layer.multiply_2d(data_df3, data_df4)
    res9_0.show(24)
    # print(np.dot(arr3, arr4))

    # test10: local to dataframe
    # print("test10:")
    # print(arr3)
    res10_0 = storage_layer.local_to_dataframe(arr3)
    res10_0.show(20)


if __name__ == '__main__':
    sl = StorageLayer()
    test_size = [5, 10, 100, 5]
    for i, size in enumerate(test_size):
        arr_df1 = sl.random_2d(size, size)
        arr_df2 = sl.random_2d(size, size)
        arr_df = sl.zeroes(size, size, size, size)
        start_time = time.time()
        print(start_time)
        print("start multiply...")
        res = sl.multiply_2d(arr_df1, arr_df2)
        print("end multiply...")
        # print("result_line_count: %d." % res.count())
        print('multiply %d * %d spend time: %f s.' % (size, size, (time.time() - start_time)))

        start_time = time.time()
        arr_df_reverse = sl.reverse(arr_df, 1)
        print('reverse %d * %d spend time: %f s.' % (size, size, (time.time() - start_time)))
        # res.show(10)