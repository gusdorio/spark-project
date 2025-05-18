from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Test Application") \
    .getOrCreate()

# Create a simple dataframe
data = [("John", 30), ("Alice", 25), ("Bob", 35)]
df = spark.createDataFrame(data, ["Name", "Age"])

# Display the dataframe
print("Data:")
df.show()

# Simple transformation
older_than_30 = df.filter(df.Age > 30)
print("People older than 30:")
older_than_30.show()

# Close the session
spark.stop()
