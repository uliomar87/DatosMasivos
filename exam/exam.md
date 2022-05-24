# Exam Unit2 big data
```scala
Import libraries and spark session

// Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifier").getOrCreate()


//1. Load iris dataframe and data clean
// Load iris dataframe
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("C:/Users/Ulipro87/Desktop/iris.csv")

// Null fields are removed
val dataClean = data.na.drop()

//2. Show columns name
dataClean.columns

//3. Print data schema
dataClean.printSchema()

//4. Print the first five columns 
dataClean.show(5) 
dataClean.select($"sepal_length",$"sepal_width",$"petal_length",$"petal_width",$"species").show(5) 

//5. Use the describe () method to learn more about the data in the DataFrame. 
dataClean.describe().show()

//6. Transformation for the categorical data which will be our labels to be classified.
// A vector is declared that transforms the data to the variable "features" 
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

// Features are transformed using the dataframe 
val features = vectorFeatures.transform(dataClean)

// A "StringIndexer" is declared that transforms the data in "species" into numeric data 
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

// Adjust the indexed species with the vector features 
val dataIndexed = speciesIndexer.fit(features).transform(features)

//7. Build the classification model.
// With the variable "splits" we make a random cut 
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

// The variable "train" is declared which will have 60% of the data 
val train = splits(0)

// The variable "test" is declared which will have 40% of the data 
val test = splits(1)

// Set layer settings for the model 
val layers = Array[Int](4, 5, 4, 3)

// The Multilayer algorithm trainer is configured with its respective parameters 
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// The model is trained with the training data 
val model = trainer.fit(train)

// The model is tested already trained 
val result = model.transform(test)

//8. Print the model results.
// The prediction and the label that will be saved in the variable are selected 
val predictionAndLabels = result.select("prediction", "label")

// Some data of the prediction is shown against the real ones to see results 
predictionAndLabels.show(50)

//The estimation of the precision of the model is executed 
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")

```
