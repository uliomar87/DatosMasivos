<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>

<H2><p align="Center">TECNOLÓGICO NACIONAL DE MÉXICO</p></H2>

<H2><p align="Center">INSTITUTO TECNOLÓGICO DE TIJUANA</p></H2>

<H2><p align="Center">SUBDIRECCIÓN ACADÉMICA</p></H2>

<H2><p align="Center">DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN</p></H2>

<H2><p align="Center">NOMBRE DE LOS ALUMNOS: </p></H2>

<H2><p align="Center">MADRIGAL RAMOS ULISES OMAR 18210496</p></H2>

<H2><p align="Center">PEREZ MORA ANA IVONNE 18212074 </p></H2>

<H2><p align="Center">Carrera: Ing. Sistemas computacionales</p></H2>

<H2><p align="Center">MATERIA: Minería de datos</p></H2>

<H2><p align="Center">PROFESOR: JOSE CHRISTIAN ROMERO HERNANDEZ</p></H2>

<H2><p align="Center">TRABAJO: Final project</p></H2>


<br>
<br>
<br>
<br>


# Unit-4 Final Project

## Introduction
The collection and analysis of data is one of the most constant practices today, helping with the implementation of Big Data, it is known that the results that are collected are often very high, this occurs due to the data that moves and that, in turn, time they carry the content of large dimensions to be able to analyze. That is why the comparison of 4 classification algorithms is made, SVM, Decision Three, Logistic Regression, Multilayer perceptron in which we can understand the behavior of each algorithm and observe which of them is the most efficient. For the execution of these algorithms, the tool that we will use will be Spark-Scala in order to manage the data together with the algorithms. 

# Theoretical framework of algorithms
## Support vector machine (SVM)
A support vector machine (SVM) is a supervised learning algorithm that can be used for binary classification or regression. Support vector machines are very popular in applications such as natural language processing, speech, image recognition, and computer vision.
A support vector machine constructs an optimal hyperplane in the form of a surface
such that the margin of separation between the two classes in the data is
widen to the maximum. Support vectors refer to a small subset of the
training observations that are used as support for the optimal location of the decision surface.
Support vector machines belong to a class of learning algorithms
called kernel methods and are also known as core machines.
nucleus.

<img alt="Imagen 1" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/1.PNG?raw=true">

Support vector machines belong to a class of machine learning algorithms called kernel methods and are also known as kernel machines.

The training of a support vector machine consists of two phases:

● Transform the predictors (input data) into a highly dimensional feature space. At this stage it is enough to specify the kernel; data is never explicitly transformed to feature space. This process is known exclusively as the kernel hack.
<br>

● Solve a quadratic optimization problem that fits an optimal hyperplane to classify the transformed features into two classes. The number of transformed features is determined by the number of support vectors.

## Decision tree
The decision tree is the most powerful and popular classification and prediction tool.
A decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents a test result, and each leaf node (terminal node) has a class label.
Decision trees classify instances by ranking them up the tree from the root to some leaf node, which gives the instance ranking. An instance is classified by starting at the root node of the tree, testing the attribute specified by this node, then moving down the branch of the tree corresponding to the value of the attribute as shown in the figure above. This process is then repeated for the subtree rooted in the new node.

<img alt="Imagen 2" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/2.PNG?raw=true">

The strengths of decision tree methods are:<br>
● Decision trees can generate understandable rules.<br>
● Decision trees perform classification without requiring much computation.<br>
● Decision trees can handle continuous and categorical variables.<br>
● Decision trees provide a clear indication of which fields are most
important for prediction or classification.

## Logistic regression
Logistic Regression is a Machine Learning algorithm that is used for classification problems, it is a predictive analysis algorithm and is based on the concept of probability.

<img alt="Imagen 3" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/3.PNG?raw=true">

We can call a logistic regression a linear regression model, but the regression
logistics uses a more complex cost function, this cost function can be defined
as the "sigmoid function" or also known as the "logistic function" instead of a
lineal funtion.
The logistic regression hypothesis tends to limit the cost function between 0 and 1. Therefore,
linear functions do not represent it since it can have a value greater than 1 or less
than 0, which is not possible according to the logistic regression hypothesis.
What is the sigmoid function?
To map predicted values to probabilities, we use the Sigmoid function. The
function maps any real value to another value between 0 and 1. In machine learning,
we use sigmoid to assign predictions to probabilities.

<img alt="Imagen 4" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/4.PNG?raw=true">

## Multilayer perceptron
The perceptron is very useful for classifying data sets that can be separated
linearly. They run into serious limitations with data sets that do not fit this pattern as discovered with the XOR problem. The XOR problem shows that for any four-point classification there exists a set that are not linearly separable.
MultiLayer Perceptron (MLP) breaks this restriction and classifies data sets that are not linearly separable. They do this by using a more robust and complex architecture to learn regression and classification models for difficult data sets.
How does a multilayer perceptron work?
The Perceptron consists of an input layer and an output layer that are completely
connected. MLPs have the same input and output layers, but can have
multiple hidden layers between the aforementioned layers, as seen below.

<img alt="Imagen 5" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/5.PNG?raw=true">

The input layer consists of neurons that accept input values. The output of these neurons is the same as that of the input predictors. The input layer nodes represent the input data. All other nodes map inputs to outputs by linearly combining the inputs with the node's weights w and bias b by applying an activation function. This can be written in matrix form for MLPC with layers K + 1 as follows:

<img alt="Imagen 6" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/6.PNG?raw=true">

The hidden layers are located between the input and output layers. Typically, the number of hidden layers varies from one to many. It is the core computation layer that has the functions that map the input to the output of a node. The nodes of the intermediate layers use the sigmoid (logistic) function, as follows

<img alt="Imagen 7" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/7.PNG?raw=true">

The output layer is the final layer of a neural network that returns the result to the user's environment. Based on the design of a neural network, it also indicates to the previous layers how they have performed in learning the information and consequently.

# Implementation of algorithms

To carry out this project, Apache Spark was used with the programming language
Scala, it was decided to use this Framework for its efficiency for Big Data, in addition to the fact that it can be used with 3 different programming languages (Python, Java and Scala), Scala was used due to the many advantages it offers such as the easy scalability of the code , is a language based on the Object Oriented paradigms and the Functional paradigm, it can be executed in Java Virtual Machine, it is also faster than Python and Java. Another advantage that we could find in Apache Spark is the documentation of this platform that allowed us to resolve doubts throughout the development of this project.

## Code used:

### ●SVM
```scala
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

```
### ●Decision tree
```scala
//  Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

//  Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

//  Load the data stored in LIBSVM format as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("C:/Users/Ulipro87/Desktop/bank-full.csv")

//  Process of categorizing the variables type string to numeric. 
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = no.withColumn("y",'y.cast("Int"))

//  Vector is created with the column features 
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

//  Transforms into a new df 
val data2 = assembler.transform(newcolumn)

//  Column and label are given a new name 
val featuresLabel = data2.withColumnRenamed("y", "label")

//  Select index
val dataIndexed = featuresLabel.select("label","features")
// Index columns

//  Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)

//  Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
// features with > 4 distinct values are treated as continuous.

//  Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

//  Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

//  Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//  Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

//  Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

//  Make predictions.
val predictions = model.transform(testData)

//  Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

//  Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

```
### ●Logistic Regression

```scala
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

```
### ●Multilayer perceptron
```scala
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
```

# Results
<img alt="Imagen 8" src="https://github.com/uliomar87/DatosMasivos/blob/unit4/Exam/img/8.PNG?raw=true">



# Conclusion
Today, multiple amounts of very valuable information are generated to interpret and analyze behaviors for different areas such as commerce, so we have to use algorithms to carry out Big Data as a fundamental tool in data analysis, since with this we can create large information networks to interpret and search through thousands of data.
With the realization of this project we observe which is the best option to analyze our data using the Spark and Scala tool, which are the next union for the interpretation of the data.

# References
● Joaquín Amat Rodrigo. (Abril 2017). Máquinas de Vector Soporte (Support Vector Machines, SVMs). 5 Junio del 2022, de Ciencia de datos.net Sitio web: Máquinas de Vector Soporte (Support Vector Machines, SVMs) <br>
● HAdolfo Sánchez Burón. (23 de diciembre del 2020). Support Vector Machine con kernlab. 5 de Junio del 2022, de ML2Projects Sitio web: Support Vector Machine con kernlab <br>
● Oliva Zhao. (24 Marxo del 2021). Algoritmos de aprendizaje automático: árboles de decisión. 5 de Junio del 2022, de Huawei Sitio web: Desicion Tree Clasifier <br>
●Himanshu Rajput. (28 Marzo del 2018). MachineX: simplificación de la regresión logística. 5 Junio del 2022, de knóldus Sitio web: MachineX: Simplifying Logistic Regression - Knoldus Blogs <br>
● Simran Kaur. (Enero 2022). Regresión logística usando PyTorch. 5 Junio del 2022, de linuxhint Sitio web: Logistic Regression  
● Jose Antonio Mora . (5 Octubre 2015). MULTILAYER PERCEPTRON. 5 Junio del 2022, Sitio web: Multilayer Perceptron | José Antonio Mora <br>
● Robert Keim. (26 Diciembre 2016). How to Train a Multilayer Perceptron Neural Network. 5 Junio del 2022, de All About Circuits Sitio web: How to Train a Multilayer Perceptron Neural Network - Technical Articles 
