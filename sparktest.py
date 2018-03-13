"""SimpleApp"""
import os

os.environ["SPARK_HOME"] = "/home/vicky/spark-2.3.0-bin-hadoop2.7/"
os.environ["PYSPARK_PYTHON"] = "/home/vicky/anaconda3/bin/python"


from pyspark import SparkContext

logFile = "/home/vicky/spark-2.3.0-bin-hadoop2.7/README.md"
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print("Lines with a: %i, lines with b: %i"%(numAs, numBs))

