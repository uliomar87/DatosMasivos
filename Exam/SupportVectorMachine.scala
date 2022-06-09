import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer,VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

// Possible mistakes
Logger.getLogger("org").setLevel(Level.ERROR)

// Session spark
val spark = SparkSession.builder().getOrCreate()

// Load dataset
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// To see data types
// df.printSchema()
// df.show(1)

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

// We modify the column "y" which is the output variable
// this indicates if the client will sign a term deposit
// how it will be classified based on this it has to be converted to numeric
// stringindexer will create a new column with the values ​​of "and" but in numeric
// being "0.0" for "no" and "1.0" for "yes"

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")

// Algorithm svm accuracy //

// We divide the data into an array into parts of 70% and 30%
val Array(training, test) = df.randomSplit(Array(0.7, 0.3), seed = 11L)

// We use linearSVC with the fectures and the label of our dataset
val lsvc = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(10).setRegParam(0.1)

// A new pipeline is created with the elements: labelIndexer,assembler,lsvc
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,lsvc))

// Fit the model
val model = pipeline.fit(training)

// Results are taken in the Test set with transform
val result = model.transform(test)

// Results in the set Test with transform
val predictionAndLabels = result.select("prediction", "label")

// Convert test results to RDD using .as and .rdd
val predictionAndLabelsrdd = result.select($"prediction",$"label").as[(Double, Double)].rdd

println("\nAlgorithm Linear Support Vector Machine Accuracy\n")

// Initialize a MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabelsrdd)

// Print algorithm accuracy
println("Accuracy:")
println(metrics.accuracy)