
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

## Create a nice data structure and load it in with sensible names!
spark = SparkSession.builder.appName("ML-IRIS").getOrCreate()

schema = StructType([StructField("sepal_length", DoubleType(), True),
                     StructField("sepal_width", DoubleType(), True),
                     StructField("petal_length", DoubleType(), True),
                     StructField("petal_width", DoubleType(), True),
                     StructField("species", StringType(), True)])

# load the data
data = spark.read.csv('./dataset/iris.data', header=False, schema=schema)

data.printSchema()

data.select("sepal_length").show(5)

## Start looking at a random forest classifier

numeric_features = [t[0] for t in data.dtypes if t[1] == 'double']

print("Prediction Features")
print(numeric_features)

test1 = data.select(numeric_features).describe().toPandas().transpose()
print("line 30, test1:")
print(test1)

## Some fun stuff
## The VectorAssembler apparently turns tables in vector entries within the same table (at least potentially)

from pyspark.ml.feature import VectorAssembler

numericCols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
df = assembler.transform(data)
print("line 42, df.show:")
df.show()

## The String Indexer is a cheap way of getting label numbers out of string

from pyspark.ml.feature import StringIndexer

label_stringIdx = StringIndexer(inputCol='species', outputCol='labelIndex')
df = label_stringIdx.fit(df).transform(df)
print("line 51, df.show:")
df.show()

## Split the datasets into groups

train, test = df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: ", str(train.count()))
print("Test Dataset Count: ", str(test.count()))

## Random Forest Time!!

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'labelIndex')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'labelIndex', 'rawPrediction', 'prediction', 'probability').show(25)

predictions.select('labelIndex', 'prediction').show(10)

## Evaluator

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))


## Separate out the stats to consider multiple classes (i.e., confusion matrix)

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds_and_labels = predictions.select(['prediction','labelIndex']).withColumn('labelIndex', F.col('labelIndex').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','labelIndex'])
print("\n")
print(preds_and_labels)
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())