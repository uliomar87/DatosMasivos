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

<H2><p align="Center">MATERIA: Big data </p></H2>

<H2><p align="Center">PROFESOR: JOSE CHRISTIAN ROMERO HERNANDEZ</p></H2>

<H2><p align="Center">TRABAJO: Exam unit 3 </p></H2>


<br>
<br>
<br>
<br>


# Exam unit 3

## Code 
```scala
// We import the library to start spark  
import org.apache.spark.sql.SparkSession 

// Use lines of code to minimize errors.
import org.apache.log4j._  
Logger.getLogger("org").setLevel(Level.ERROR)  

// Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()  

// Import the Kmeans library for the clustering algorithm.
import org.apache.spark.ml.clustering.KMeans 

// We load the dataset of wholesale customer data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("C:/Users/Ulipro87/Desktop/WholesaleCustomersData.csv")  
dataset.show  
```
Result:<br>
<img alt="Imagen 1" src="">


```scala
// Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data 
val  feature_data  = dataset.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")  
feature_data.show  
```
Result:<br>
<img alt="Imagen 2" src="">


```scala
// Import Vectorassembler and Vector
import org.apache.spark.ml.feature.VectorAssembler  

// We create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels. 
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper",  "Delicassen")).setOutputCol("features")  

// Use the assembler object to transform feature_data 
val  features = assembler.transform(feature_data)  
features.show  
val kmeans = new KMeans().setK(3).setSeed(1L) 
val model = kmeans.fit(features)  
```
Result:<br>
<img alt="Imagen 3" src="">


```scala
// We evaluate the clusters using the sum of squared errors within the WSSSE set and print the centroids. 
val WSSSE = model.computeCost(features)  
println(s"Within Set Sum of Squared Errors = $WSSSE")  
```
Result:<br>
<img alt="Imagen 4" src="">

```scala
// Group Printing Centers
println("Cluster Centers: ")  
model.clusterCenters.foreach(println)
```
Result:<br>
<img alt="Imagen 5" src="">
