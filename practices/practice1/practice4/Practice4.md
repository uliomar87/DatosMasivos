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

<H2><p align="Center">TRABAJO: Practice 4 - Gradient-boosted tree classifier</p></H2>


<br>
<br>
<br>
<br>

# Practice 4 - Gradient-boosted tree classifier

Gradient-boosted trees (GBTs) are a popular classification and regression method using ensembles of decision trees.
The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.


# Code:
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)
  .setFeatureSubsetStrategy("auto")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray(0))

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

```
Resultado: 

```scala
scala> println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
Learned classification GBT model:
 GBTClassificationModel (uid=gbtc_2569658fe919) with 10 trees
  Tree 0 (weight 1.0):
    If (feature 406 <= 126.5)
     If (feature 99 in {2.0})
      Predict: -1.0
     Else (feature 99 not in {2.0})
      Predict: 1.0
    Else (feature 406 > 126.5)
     Predict: -1.0
  Tree 1 (weight 0.1):
    If (feature 406 <= 126.5)
     If (feature 549 <= 253.5)
      If (feature 541 <= 162.5)
       Predict: 0.47681168808847024
      Else (feature 541 > 162.5)
       Predict: 0.4768116880884703
     Else (feature 549 > 253.5)
      Predict: -0.4768116880884694
    Else (feature 406 > 126.5)
     If (feature 127 <= 140.0)
      Predict: -0.4768116880884701
     Else (feature 127 > 140.0)
      Predict: -0.4768116880884712
  Tree 2 (weight 0.1):
    If (feature 406 <= 126.5)
     If (feature 631 <= 5.5)
      Predict: -0.4381935810427206
     Else (feature 631 > 5.5)
      If (feature 524 <= 39.5)
       If (feature 126 <= 6.0)
        Predict: 0.4381935810427206
       Else (feature 126 > 6.0)
        Predict: 0.43819358104272066
      Else (feature 524 > 39.5)
       Predict: 0.4381935810427206
    Else (feature 406 > 126.5)
     If (feature 406 <= 251.5)
      Predict: -0.4381935810427206
     Else (feature 406 > 251.5)
      Predict: -0.43819358104272066
  Tree 3 (weight 0.1):
    If (feature 489 <= 37.5)
     If (feature 549 <= 253.5)
      Predict: 0.4051496802845983
     Else (feature 549 > 253.5)
      Predict: -0.4051496802845982
    Else (feature 489 > 37.5)
     If (feature 97 in {0.0})
      If (feature 404 <= 21.0)
       Predict: -0.4051496802845983
      Else (feature 404 > 21.0)
       Predict: -0.40514968028459836
     Else (feature 97 not in {0.0})
      Predict: -0.4051496802845982
  Tree 4 (weight 0.1):
    If (feature 433 <= 66.5)
     If (feature 100 <= 193.5)
      If (feature 573 <= 11.5)
       If (feature 597 <= 202.0)
        Predict: 0.3765841318352991
       Else (feature 597 > 202.0)
        Predict: 0.37658413183529915
      Else (feature 573 > 11.5)
       Predict: 0.37658413183529926
     Else (feature 100 > 193.5)
      Predict: -0.3765841318352994
    Else (feature 433 > 66.5)
     If (feature 459 <= 83.5)
      Predict: -0.37658413183529926
     Else (feature 459 > 83.5)
      Predict: -0.3765841318352992
  Tree 5 (weight 0.1):
    If (feature 406 <= 126.5)
     If (feature 157 <= 253.5)
      Predict: 0.35166478958101
     Else (feature 157 > 253.5)
      Predict: -0.3516647895810099
    Else (feature 406 > 126.5)
     Predict: -0.35166478958100994
  Tree 6 (weight 0.1):
    If (feature 434 <= 79.5)
     If (feature 295 <= 253.5)
      If (feature 262 <= 84.0)
       Predict: 0.32974984655529926
      Else (feature 262 > 84.0)
       Predict: 0.3297498465552993
     Else (feature 295 > 253.5)
      Predict: -0.32974984655530015
    Else (feature 434 > 79.5)
     If (feature 349 <= 68.5)
      Predict: -0.32974984655529926
     Else (feature 349 > 68.5)
      Predict: -0.3297498465552993
  Tree 7 (weight 0.1):
    If (feature 406 <= 126.5)
     If (feature 628 <= 0.5)
      Predict: -0.3103372455197954
     Else (feature 628 > 0.5)
      If (feature 237 <= 221.5)
       If (feature 152 <= 2.5)
        Predict: 0.3103372455197956
       Else (feature 152 > 2.5)
        Predict: 0.3103372455197957
      Else (feature 237 > 221.5)
       Predict: 0.31033724551979575
    Else (feature 406 > 126.5)
     Predict: -0.3103372455197956
  Tree 8 (weight 0.1):
    If (feature 406 <= 126.5)
     If (feature 627 <= 2.5)
      Predict: -0.2930291649125432
     Else (feature 627 > 2.5)
      If (feature 598 <= 27.5)
       Predict: 0.2930291649125433
      Else (feature 598 > 27.5)
       If (feature 598 <= 112.0)
        Predict: 0.2930291649125433
       Else (feature 598 > 112.0)
        Predict: 0.2930291649125434
    Else (feature 406 > 126.5)
     If (feature 350 <= 233.5)
      If (feature 242 <= 70.5)
       Predict: -0.2930291649125433
      Else (feature 242 > 70.5)
       Predict: -0.2930291649125434
     Else (feature 350 > 233.5)
      Predict: -0.2930291649125434
  Tree 9 (weight 0.1):
    If (feature 489 <= 37.5)
     If (feature 344 <= 253.5)
      If (feature 328 <= 78.5)
       Predict: 0.2775066643835825
      Else (feature 328 > 78.5)
       Predict: 0.2775066643835826
     Else (feature 344 > 253.5)
      Predict: -0.27750666438358174
    Else (feature 489 > 37.5)
     If (feature 405 <= 207.5)
      If (feature 349 <= 2.0)
       Predict: -0.27750666438358246
      Else (feature 349 > 2.0)
       Predict: -0.27750666438358257
     Else (feature 405 > 207.5)
      Predict: -0.27750666438358257

```

