1. Start a simple Spark session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

2. Load the Netflix Stock CSV file, have Spark infer the data types.
val NetflixDf = spark.read.option("header", "true").option("inferSchema",
  "true")csv("Netflix_2011_2016.csv")

3. What are the column names? 
NetflixDf.columns


4. How is the scheme?
NetflixDf.printSchema()


5. Print the first 5 columns.
NetflixDf.select($"Date",$"Open",$"High",$"Low", $"Close").show(5)

6. Use describe() to learn about the DataFrame. 
NetflixDf.describe().show()

7. Create a new dataframe with a new column called “HV Ratio” which is the ratio of the price in the “High” column to the “Volume” column of shares traded for one day. Hint - is an operation
var NewDf = NetflixDf.withColumn("HV Ratio", NetflixDf("High")/NetflixDf("Volume")).show()

8. Which day had the highest peak in the “Open” column? 
val DiaDf = NetflixDf.withColumn("Day", dayofmonth(NetflixDf("Date")))
val MaxDf = DiaDf.groupBy("Day").max()
MaxDf.printSchema()
MaxDf.select($"Day", $"max(Open)").show()
MaxDf.select($"Day", $"max(Open)").sort(desc("max(Open)")).show()
MaxDf.select($"Day", $"max(Open)").sort(desc("max(Open)")).show(1)

9. What is the meaning of the “Close” column in the context of financial information,
explain it, there is no need to code anything?
What close does is describe the average with which the day was closed.

10. What is the maximum and minimum of the “Volume” column?
NetflixDf.select(max("Volume"), min("Volume")).show()

11. With Scala/Spark Syntax $ answer the following:
a. How many days was the “Close” column under $600?
NetflixDf.filter($"Close" < 600).count()

b. What percentage of the time was the "High" column greater than $500?
val dias = NetflixDf.filter($"High" > 500).count().toDouble
val porcentaje = ((dias / NetflixDf.count())*100) 

c. What is the Pearson correlation between the “High” column and the “Volume” column?
NetflixDf.select(corr($"High", $"Volume")).show()

d. What is the maximum of the “High” column per year?
val MaxHDf = NetflixDf.withColumn("year", year(NetflixDf("Date")))
val MaxHPDf= MaxHDf.groupBy("year").max()
MaxHPDf.select($"year", $"max(High)").show()

e. What is the average of the “Close” column for each calendar month?
val MesDf = NetflixDf.withColumn("Month", month(NetflixDf("Date")))
val MeanDf = MesDf.groupBy("Month").mean()
MeanDf.select($"Month", $"avg(Close)").sort(asc("Month")).show()