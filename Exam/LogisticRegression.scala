// Import libraries
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

//  Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.getOrCreate()

// Load the data stored in LIBSVM format as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("C:/Users/Ulipro87/Desktop/bank-full.csv")

// Process of categorizing variables from string to numeric type.
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = no.withColumn("y",'y.cast("Int"))

// The vector is created with the characteristics of the column
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

// Transform into a new df
val data2 = assembler.transform(newcolumn)

// Column and label get a new name
val featuresLabel = data2.withColumnRenamed("y", "label")

// Select index
val dataIndexed = featuresLabel.select("label","features")

// Split the data into treaning and test sets
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

//  Logistic regression
val logisticReg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val model = logisticReg.fit(trainingData)
val predictions = model.transform(testData)
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Print results
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error: ${(1.0 - metrics.accuracy)}")