//  Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, VectorAssembler}

//  Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

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

// Split the data into train and test
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1) 

// Specify the layers for the neural network:
// input layer of size 5 (features), two intermediate ones of size 2 and 2
// and output of size 4 (classes)
val layers = Array[Int](5,2,2,4)

// Create the trainer and configure its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Train the model
val model = trainer.fit(train)

// Calculate the precision on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")