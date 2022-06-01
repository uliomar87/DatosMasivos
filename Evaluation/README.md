# We import the library to start spark  
import org.apache.spark.sql.SparkSession  
# Use lines of code to minimize errors.
import org.apache.log4j._  
Logger.getLogger("org").setLevel(Level.ERROR)  
# Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()  
# Import the Kmeans library for the clustering algorithm.
import org.apache.spark.ml.clustering.KMeans  
# We load the dataset of wholesale customer data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("/home/ivonne/Descargas/Wholesale.csv")  
dataset.show  
# Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data 
val  feature_data  = dataset.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")  
feature_data.show  
# Import Vectorassembler and Vector
import org.apache.spark.ml.feature.VectorAssembler  
# We create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels. 
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper",  "Delicassen")).setOutputCol("features")  
# Use the assembler object to transform feature_data 
val  features = assembler.transform(feature_data)  
features.show  
val kmeans = new KMeans().setK(3).setSeed(1L) 
val model = kmeans.fit(features)  
# We evaluate the clusters using the sum of squared errors within the WSSSE set and print the centroids. 
val WSSSE = model.computeCost(features)  
println(s"Within Set Sum of Squared Errors = $WSSSE")  
# Group Printing Centers
println("Cluster Centers: ")  
model.clusterCenters.foreach(println)  