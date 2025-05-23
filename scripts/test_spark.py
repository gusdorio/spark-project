from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml import Pipeline
import os

spark = SparkSession.builder \
    .appName("PCA-Architecture-Demo") \
    .master("local[4]") \
    .getOrCreate()

# Create DataFrame with random columns first
df_random = spark.range(1000).select(
    rand(seed=42).alias("col1"),
    rand(seed=43).alias("col2"), 
    rand(seed=44).alias("col3")
)

# Assemble into vector
assembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
df = assembler.transform(df_random).select("features")

# Determine partitions
num_partitions = int(os.getenv("SPARK_NUM_PARTITIONS", spark.sparkContext.defaultParallelism))
df = df.repartition(num_partitions)

# PCA Pipeline
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df)

# Results
results = pca_model.transform(df).select("pca_features")
results.show(5)

spark.stop()