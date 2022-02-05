"""

The 3V's of Big Data

* Volume, Variaty and Velocity

* Volume: Size of the data

* Variaty: Different sources and formats


* Velocity: Speed of the data

Big Data concepts and Terminology

* Clustered computing: Collection of resources of multiple machines

* Parallel computing: Simultaneous computation

* Distributed computing: Collection of nodes(networked computers) tha run in parallel
* Batch processing: Breaking the job into small pieces and running them on individual machines


* Real-time processing: Immediate processing of data


Apache Spark: General purpose and lightning fast computing system
- Open sorce
-Both batch and real-time data processing




Features of Apache Spark framework
* Distributed cluster computing framework

* Efficient in-memory computations for large data setes

* Lightning fast data processing framework

* Provides support for Java, Scala, Python, R and SQL


Spark offer features like machine learning, real-time stream processing, and graph computations

Spark Components

1-Spark SQL
2-MLlib Machine Learning

3- GraphX

4- Spark Streaming

Todas estas estan sobre el Apache Spark Core RDD API


Spark modes of deployment
* Local mode: Single machine such as your laptop
		* Local model convenint for testing, debugging and demostration

* Cluster mode: Set of pre-defined machines
		* Good for production

* Workflow: Local -> clusters

* No code change necessary

Overiew of PySpark

* Apache Spark is written in Scala
* To support Python with Spark, APache Spark community released PySpark

* similar computation speed and power as Scala

* PySpark APIs are similar to Pandas and Scikit-learn


What is Spark shell?

* Interactive environment for running Spark jobs

* Helpful for fast interactive prototyping

* Spark's shells allow interacting with data on disk or in memory

* Three different Spark shells:
		* Spark-shell for Scala
		* PySpark-shell for Python
		* SparkR for R


PySpark shell

* PySpark shell is the Python-based command line tool

* PySpark shell allow data scientists interface with Spark data structures

* PySpark shell support connecting to a cluster 

Understanding SparkContext

* SparkContext is an entry point into the world of Spark

* An entry point is a way of connecting to Spark cluster

* An entry points is like a key to the house

* PySpark has a dafault SparkContext  called sc

sc.version

sc.pythonVer

sc.master


Loading data in PySpark

 rdd = sc.parallelize([1, 2, 3, 4, 5])

 rdd2 = sc.textFile("test.txt")

 ------------------------------------------

 Understanding SparkContext
A SparkContext represents the entry point to Spark functionality. It's like a key to your car. When we run any Spark application, a driver program starts, which has the main function and your SparkContext gets initiated here. PySpark automatically creates a SparkContext for you in the PySpark shell (so you don't have to create it by yourself) and is exposed via a variable sc.

In this simple exercise, you'll find out the attributes of the SparkContext in your PySpark shell which you'll be using for the rest of the course.

"""
# Print the version of SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)

# Print the Python version of SparkContext
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)

# Print the master of SparkContext
print("The master of Spark Context in the PySpark shell is", sc.master)



"""
Interactive Use of PySpark
Spark comes with an interactive Python shell in which PySpark is already installed in it. PySpark shell is useful for basic testing and debugging and it is quite powerful. The easiest way to demonstrate the power of PySpark’s shell is to start using it. In this example, you'll load a simple list containing numbers ranging from 1 to 100 in the PySpark shell.

The most important thing to understand here is that we are not creating any SparkContext object because PySpark automatically creates the SparkContext object named sc, by default in the PySpark shell.


"""


# Create a Python list of numbers from 1 to 100 
numb = range(1, 100)

# Load the list into PySpark  
spark_data = sc.parallelize(numb)




"""
Loading data in PySpark shell
In PySpark, we express our computation through operations on distributed collections that are automatically parallelized across the cluster. In the previous exercise, you have seen an example of loading a list as parallelized collections and in this exercise, you'll load the data from a local file in PySpark shell.

Remember you already have a SparkContext sc and file_path variable (which is the path to the README.md file) already available in your workspace.


"""


# Load a local file into PySpark shell
lines = sc.textFile(file_path)




"""
Use of Lambda 
function in python-filter()


What are anonymous functions in Python?

* Lambda functions are anontmous functions in Python

* Very powerful and used in Python. Quite efficient with map() and filter()

* Lambda functions create functions to be called later similar to def

* it returnste functions without any name 

* Inline a function definition or to defer execution of a code


Lambda function syntax

lambda arguments: expression

* Example of lambda function

double = lambda x: x * 2

print(double(3))


Difference between def vs lambda functions

* Python code to illustrate cube of a number


def cube(x):
	return x ** 3
g = lambda x: x ** 3

print(g(10))

print(cube(10))


Use of Lambda function in Python - map()

* map() function takes a function and a list and returns a new list which contains items
returned by that function for each item

* General syntax of map()

map(function, list)

* Example of map()

items = [1, 2, 3, 4]

list(map(lambda x: x + 2, items))


Use of Lambda function in python - filter()

* filter() function takes a function and a list and returns a new list for which the function evaluates as true

* General syntax of filter()


* Example of filter()

items = [1, 2, 3, 4]

list(filter(lambda x: (x%2 ! = 0), items))

"""



"""
Use of lambda() with map()
The map() function in Python returns a list of the results after applying the given function to each item of a given iterable (list, tuple etc.). The general syntax of map() function is map(fun, iter). We can also use lambda functions with map(). The general syntax of map() function with lambda() is map(lambda <argument>:<expression>, iter). Refer to slide 5 of video 1.7 for general help of map() function with lambda().

In this exercise, you'll be using lambda function inside the map() built-in function to square all numbers in the list.
"""
# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list
squared_list_lambda = list(map(lambda x:  x * x, my_list))
squared_list_lambda_2 = list(map(lambda x:  x ** 2, my_list))
# Print the result of the map function
print("The squared numbers are", squared_list_lambda)

print("The squared numbers are using 2", squared_list_lambda_2)


"""
Use of lambda() with filter()
Another function that is used extensively in Python is the filter() function. The filter() function in Python takes in a function and a list as arguments. The general syntax of the filter() function is filter(function, list_of_input). Similar to the map(), filter() can be used with lambda function. The general syntax of the filter() function with lambda() is filter(lambda <argument>:<expression>, list). Refer to slide 6 of video 1.7 for general help of the filter() function with lambda().

In this exercise, you'll be using lambda() function inside the filter() built-in function to find all the numbers divisible by 10 in the list.


"""

# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)


"""
What is RDD?

RDD stands for Resilient Distributed Datasets, it is simply a collection of data distributed across the cluster.
RDD is the fundamental and backbone data type in PySpark


When Spark starts processing data, it divides the data into partitions and distributes the data across cluster nodes, with each node containing a slice of data.

different features of RDD


Decomposing RDDs

* Resilient Distributed Datasets

* Resilient: Ability to withstand failures and recompute missing or damaged partitions

* Distributed: Spanning across multiple machines, or nodes in the cluster for efficient computation


* Datasets: Collection of partitioned data e.g Arrays, Tables, Tuples etc ...

Creating RDDs. How to do it?

* Parallelizing an existing collection of objects

* External dataset:
* Files in HDFS
* Objects in Amazon S3 bucket

* lines in a text file

* From existing RDDs


* Parallelized collection(parallelizing)

* parallelize() for creating RDDs from python lists


numRDD = sc.parallelize([1, 2, 3, 4])

helloRDD = sc.parallelize("Hello world")

type(helloRDD)

From external datasets

* textFile() for creating RDDs from external datasets

fileRDD = sc.textFile("README.md")

type(fileRDD)


Understanding Partitioning in PySpark

understanding how Spark deals wit partitions allow one to control parallelism

Understanding Partitioning in PySpark

* A partition is a logical dvision of a large distributed data set with each part being stored in multiple location across the cluster



By default Spark partitions the data athe time of creating RDD based on several factor such as 

available resources, external datasets etc, however, this behavior can be controlled by passing a second argument called minPartitions which defines the minimum number of prtition to be created for an RDD 


numRDD = sc.parallelize(range(10), minPartitions= 6)

fileRDD = sc.textFile("README.md", minPartitions = 6)

The number of partitions in an RDD can be found by using getNumPartitions() method

"""

"""
RDDs from Parallelized collections
Resilient Distributed Dataset (RDD) is the basic abstraction in Spark. It is an immutable distributed collection of objects. Since RDD is a fundamental and backbone data type in Spark, it is important that you understand how to create it. In this exercise, you'll create your first RDD in PySpark from a collection of words.

Remember you already have a SparkContext sc available in your workspace.

"""
# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])


# Print out the type of the created object
print("The type of RDD is", type(RDD))


"""


RDDs from External Datasets
PySpark can easily create RDDs from files that are stored in external storage devices such as HDFS (Hadoop Distributed File System), Amazon S3 buckets, etc. However, the most common method of creating RDD's is from files stored in your local file system. This method takes a file path and reads it as a collection of lines. In this exercise, you'll create an RDD from the file path (file_path) with the file name README.md which is already available in your workspace.

Remember you already have a SparkContext sc available in your workspace.

"""
# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))


"""
Partitions in your data
SparkContext's textFile() method takes an optional second argument called minPartitions for specifying the minimum number of partitions. In this exercise, you'll create an RDD named fileRDD_part with 5 partitions and then compare that with fileRDD that you created in the previous exercise. Refer to the "Understanding Partition" slide in video 2.1 to know the methods for creating and getting the number of partitions in an RDD.

Remember, you already have a SparkContext sc, file_path and fileRDD available in your workspace.

"""


# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())




"""
Transformation action

Spark Operations = Transformations + Actions

* Transformations create new RDDS

* Actions perform computation on the RDDS


The most important feature which helps RDDs in fault tolerance and optimizing resource use is the lazy evaluation

Spark creates a graph from all the operations you perfom on an RDD and execution of the graph starts only when an action is performed on RDD as shown in this figure


Storage---->Rdd created by reading data from stable storage-------->RDD1--->transfromation----->RDD2--->transfromation----->RDD3--->action-->(result)

* Basic RDD Transformations

* map(), filter(), flatMap(), and union()

map() Transformation


* map() transformation applies a function to all elements in the RDD


RDD = sc.parallelize([1,2,3,4])

RDD_map = RDD.map(lambda x: x * x)


Filter() Transformation

* Filter transformation returns a new RDD with only the elements tha pass the condition


RDD = sc.parallelize([1, 2, 3, 4])

RDD_filter = RDD.filter(lambda x: x > 2)


flatMap() Transformation

* flatMap() transformation returns multiple values for each element in the original RDD

A simple usage of flatMap is splitting up an input string into words

in this case for example
["hello word","How are you"]

flatMap(x:x.split(""))

return-----["Hello", "world", "are","you"]


As you can see, even though the input RDD has 2 elements, the otput RDD now contains 5 elements

RDD = sc.parallelize(["hellow world", "how are you"])

RDD_flatmap = RDD.flatMap(lambda x: x.split(" "))

Union () TRansformation return the union of one RDD with another RDD

inputRDD = sc.textFile("logs.txt")

errorRDD = inputRDD.filter(lambda x: "error" in x.split())

warningsRDD = inputRDD.filter(lambda x: "warnings" in x.split())

combinedRDD= errorRDD.union(warningsRDD)


RDD Actions

Actions are the operations that are applied on RDDs to return a value after running a computation.

* Operation return a value after running a computation on the RDD


* Basic RDD Actions
		*collect()
		*take(N)
		*first()
		*count()



* collect() return all the elements of the dataset as an array


* take(N) returns an array with first N elements of th dataset

RDD_map.collect()

* first() print the first element of the RDD is similar to take(1)
RDD_map.first()

count(): is used to return the total number of rows/elements in te RDD
RDD_flatmap.count()

"""
"""
Map and Collect
The main method by which you can manipulate data in PySpark is using map(). The map() transformation takes in a function and applies it to each element in the RDD. It can be used to do any number of things, from fetching the website associated with each URL in our collection to just squaring the numbers. In this simple exercise, you'll use map() transformation to cube each number of the numbRDD RDD that you created earlier. Next, you'll return all the elements to a variable and finally print the output.

Remember, you already have a SparkContext sc, and numbRDD available in your workspace.

"""
# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x **3)

# Collect the results
numbers_all = cubedRDD.collect()

# Print the numbers from numbers_all
for numb in numbers_all:
	print(numb)


"""
Filter and Count
The RDD transformation filter() returns a new RDD containing only the elements that satisfy a particular function. It is useful for filtering large datasets based on a keyword. For this exercise, you'll filter out lines containing keyword Spark from fileRDD RDD which consists of lines of text from the README.md file. Next, you'll count the total number of lines containing the keyword Spark and finally print the first 4 lines of the filtered RDD.

Remember, you already have a SparkContext sc, file_path and fileRDD available in your workspace.


"""

# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in  line)

# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())

# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4): 
  print(line)



"""

Working with Pair

RDDs in PySpark


You'll learn how to work with RDDs of key/value pairs, which are a common data type required for many operations in Spark

Introduction to pair RDDs in PySpark

* Real life datasets are usually key/value pairs

* Each row is a key and maps to one or more values

* Pair RDD is a special data structure to work with this kind of datasets

* Pair RDD: key is the identifier and value is data

There are number of ways to create pair RDDs

Creating pair RDDs
* Two common ways to create pair RDDs
* From a list of key-value tuple
* From a regular RDD

* Get the data into key/value form for paired RDD

my_tuple = [('Sam', 23), ('Mary', 34), ('Peter', 25)]

pairRDD_tuple = sc.parallelize(my_tuple)


my_list = ['Sam 23', 'Mary 34', 'Peter 25']

regularRDD = sc.parallelize(my_list)


pairRDD_RDD = regularRDD.map(lambda s: (s.plit(" ")[0], s.split(" ")[1]))



Pair RDDs are still RDDs and thus use all the transformations available to regular RDDs

Transformations on Pair RDDs

* All regular transformations work on pair RDD

* Have to pass functions that operate on key value pairs rather tan on individual elements

* Examples of paired RDD Transformations


* reduceByKey(func):Combine value with the same key

* groupByKey(): Group values with the same key

* sortByKey(): Return an RDD sorted by the key


*join(): Join two pair RDDs based on their key




reduceByKey() transformation

* reduceByKey() transformation combines values with the same key

it runs several parallel operations, one for each key in the dataset.

Because datasets can have very large numbers of keys, reduceByKey is not implemented as an action

Instead, it returns a new RDD consisting of each key and the reduced value for that key


regularRDD = sc.parallelize([("Messi", 23), ("Ronaldo", 34), ("Neymar", 22), ("Messi", 24)])


pairRDD_reducebykey = regularRDD.reduceByKey(lambda x, y : x + y)

pairRDD_reducebykey.collect()


the result shows that player as key and total number of goals scored as value.
[('Neymar', 22), ('Ronaldo', 34), ('Messi', 47)]
"""


"""
One of the most popular pair RDD transformations is reduceByKey() which operates on key, value (k,v) pairs and merges the values for each key. In this exercise, you'll first create a pair RDD from a list of tuples, then combine the values with the same key and finally print out the result.

Remember, you already have a SparkContext sc available in your workspace.


"""

# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(3, 4), (3, 6), (4, 5)])
Rdd_2 = sc.parallelize([(3, 4), (3, 6), (4, 5), (4, 20), (3, 7), (3, 10)])
# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: x + y)

Rdd_Reduced_2 = Rdd_2.reduceByKey(lambda x, y: x + y)

# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

for num in Rdd_Reduced_2.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

"""
SortByKey and Collect
Many times it is useful to sort the pair RDD based on the key (for example word count which you'll see later in the chapter). In this exercise, you'll sort the pair RDD Rdd_Reduced that you created in the previous exercise into descending order and print the final output.

Remember, you already have a SparkContext sc and Rdd_Reduced available in your workspace.


"""


# Sort the reduced RDD with the key by descending order
Rdd_Reduced_Sort = Rdd_Reduced.sortByKey(ascending=False)

# Iterate over the result and retrieve all the elements of the RDD
for num in Rdd_Reduced_Sort.collect():
  print("Key {} has {} Counts".format(num[0], num[1]))


"""

Advanced RDD Actions

reduce() action

* reduce(func) action is used for aggregating the elements of a regular RDD

* The function should be commutative(changing the order of the operans does not change the result) and associative

* An exameple of reduce() action in PySpark

x = [1, 3, 4, 6]

RDD = sc.parallelize(x)

RDD.reduce(lambda x, y : x + y)

In many cases, it is not advisable to run collect action on RDDs because of the huge size of the data

in these cases, it's common to write data out to a distributed storage systems such as HDFS or Amazon S3

saveAsTextFile() action

* saveAsTextFile() action saves RDD into a text file inside a directory with each partition as a separate file

Example
RDD.saveAsTextFile("tempFile")

However, you can change it to return a new RDD that is reduced into a single partition using the coalesce()

RDD.coalesce(1).saveAsTextFile("tempFile")


Action Operations on pair RDDs

* RDD actions available for PySpark pair RDDs

* Pair RDD actions leverage the key-value data

* Few examples of pair RDD actions include
	* countByKey()
	* collectAsMap()


CountByKey() action
* countByKey() only available for type (K,V)

* countByKey() action counts the number of elements for each key
* Example of countByKey()on a simple list


rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])

for kee, val in rdd.countByKey().items():
	print(kee, val)


One thing to note is that countByKey should only be used on a dataset whose size i small enough to fit in memory

collectAsMap() action

* collectAsMap() return the key-value pairs in the RDD as a dictionary

* Example of collectAsMap() on a simple tuple

sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
"""


"""
CountingBykeys
For many datasets, it is important to count the number of keys in a key/value dataset. For example, counting the number of countries where the product was sold or to show the most popular baby names. In this simple exercise, you'll use the Rdd that you created earlier and count the number of unique keys in that pair RDD.

Remember, you already have a SparkContext sc and Rdd available in your workspace.

"""


# Count the unique keys
total = Rdd.countByKey()

# What is the type of total?
print("The type of total is", type(total))

# Iterate over the total and print the output
for k, v in total.items(): 
  print("key", k, "has", v, "counts")



"""
Create a base RDD and transform it
The volume of unstructured data (log lines, images, binary files) in existence is growing dramatically, and PySpark is an excellent framework for analyzing this type of data through RDDs. In this 3 part exercise, you will write code that calculates the most common words from Complete Works of William Shakespeare.

Here are the brief steps for writing the word counting program:

Create a base RDD from Complete_Shakespeare.txt file.
Use RDD transformation to create a long list of words from each element of the base RDD.
Remove stop words from your data.
Create pair RDD where each element is a pair tuple of ('w', 1)
Group the elements of the pair RDD by key (word) and add up their values.
Swap the keys (word) and values (counts) so that keys is count and value is the word.
Finally, sort the RDD by descending order and print the 10 most frequent words and their frequencies.
In this first exercise, you'll create a base RDD from Complete_Shakespeare.txt file and transform it to create a long list of words.

Remember, you already have a SparkContext sc already available in your workspace. A file_path variable (which is the path to the Complete_Shakespeare.txt file) is also loaded for you.


"""


# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)

# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())

# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())




"""
Remove stop words and reduce the dataset
After splitting the lines in the file into a long list of words in the previous exercise, in the next step, you'll remove stop words from your data. Stop words are common words that are often uninteresting. For example "I", "the", "a" etc., are stop words. You can remove many obvious stop words with a list of your own. But for this exercise, you will just remove the stop words from a curated list stop_words provided to you in your environment.

After removing stop words, you'll next create a pair RDD where each element is a pair tuple (k, v) where k is the key and v is the value. In this example, pair RDD is composed of (w, 1) where w is for each word in the RDD and 1 is a number. Finally, you'll combine the values with the same key from the pair RDD.

Remember you already have a SparkContext sc and splitRDD available in your workspace.

"""


# Convert the words in lower case and remove stop words from the stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))

# Count of the number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)




"""
Print word frequencies
After combining the values (counts) with the same key (word), in this exercise, you'll return the first 10 word frequencies. You could have retrieved all the elements at once using collect() but it is bad practice and not recommended. RDDs can be huge: you may run out of memory and crash your computer..

What if we want to return the top 10 words? For this, first you'll need to swap the key (word) and values (counts) so that keys is count and value is the word. After you swap the key and value in the tuple, you'll sort the pair RDD based on the key (count). This way it is easy to sort the RDD based on the key rather than the key using sortByKey operation in PySpark. Finally, you'll return the top 10 words from the sorted RDD.

"""


# Display the first 10 words and their frequencies from the input RDD
for word in resultRDD.take(10):
	print(word)

# Swap the keys and values from the input RDD
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[0]))

# Sort the keys in descending order
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)

# Show the top 10 most frequent words and their frequencies from the sorted RDD
for word in resultRDD_swap_sort.take(10):
	print("{},{}". format(word[1], word[0]))




"""

Introduction to PySpark
DataFrames


What are PySpark DataFrames?


* PySpark SQL is a Spark library for structured data. It provides more information about the structure of data and computation

* PySpark DataFrame is an immutable distributed collection of data with named columns

* Designed for processing both(e.g relational database) and semi-structured data (e.g json)

* Dataframe API is available in Python, R, Scala, and Java

* DataFrames in PySpark support both SQL queris(SELECT * from table) or expression methods(df.select())


SparkSession -  Entry point for DataFrame API


* SparkContext is the main entry point for creating RDDs

* SparkSession provides a single point of entry to interact with Spark DataFrames

* SparkSession is used to create DataFrame, register DataFrames, execute SQL queries

* SparkSession is available in PySpark shell as  spark 


Creating DataFrames in PySpark

* Two different methods of creating DataFrames in PySpark

	* From existing RDDs using SparkSession's create DataFrame() method

	* from various data sources(csv, json, txt) using SparkSession's read method

* Schema controls the data and help DataFrames to optimize queries

* Schema provides information about column name, type of data in the column, empty values, etc
iphones_RDD = sc.parallelize([
	("XS", 2018, 5.65, 2.79, 6.24),
	("XR", 2018, 5.94, 2.98, 6.84),
	("X10", 2017, 5.65, 2.79, 6.13),
	("8Plus", 2017, 6.23, 3.07, 7.12)

])

names = ['Model', 'Year', 'Height', 'Width', 'Weight']


iphones_df = spark.createDataFrame(iphones_RDD, schema=names)

type(iphones_df)


Create a DataFrame from reading a csv/json/txt


df_csv = spark.read.csv("people.csv", header=True, inferSchema=True)

df_json = spark.read.json("people.json", header=True, inferSchema=True)


df_txt = spark.read.txt("people.txt", header=True, inferSchema=True)


* Path to the file and two optional parameters

* two optional parameters
	* header=True, inferSchema=True
"""

"""
RDD to DataFrame
Similar to RDDs, DataFrames are immutable and distributed data structures in Spark. Even though RDDs are a fundamental data structure in Spark, working with data in DataFrame is easier than RDD, and so understanding of how to convert RDD to DataFrame is necessary.

In this exercise, you'll first make an RDD using the sample_list that is already provided to you. This RDD contains the list of tuples ('Mona',20), ('Jennifer',34),('John',20), ('Jim',26) with each tuple contains the name of the person and their age. Next, you'll create a DataFrame using the RDD and the schema (which is the list of 'Name' and 'Age') and finally confirm the output as PySpark DataFrame.

Remember, you already have a SparkContext sc and SparkSession spark available in your workspace.

"""
# Create an RDD from the list
rdd = sc.parallelize(sample_list)

# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema =['Name', 'Age'])

# Check the type of names_df
print("The type of names_df is", type(names_df))




"""
Loading CSV into DataFrame
In the previous exercise, you have seen a method for creating a DataFrame from an RDD. Generally, loading data from CSV file is the most common method of creating DataFrames. In this exercise, you'll create a PySpark DataFrame from a people.csv file that is already provided to you as a file_path and confirm the created object is a PySpark DataFrame.

Remember, you already have a SparkSession spark and a variable file_path (the path to the people.csv file) available in your workspace.


"""


# Create an DataFrame from file_path
people_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the type of people_df
print("The type of people_df is", type(people_df))


"""
Interacting with PySpark DataFrames

DataFrame operators in PySpark

* DataFrame operations: Transformations and Actions

* DataFrame Transformations:
* select(), filter(), groupby(), orderby(), dropDuplicates(), and withColumnRenamed()

*DataFrame Actions:
* printSchema(), head(), show(), count(), columns and describe()

Correction: printSchema() is a method for any Spark dataset/dataframe and not an action




select() and show() operations

* select() transformation subsets the columns in the DataFrame

df_id_age = test.select('Age') # return a new dataframe


show() action print first 20 rows in the DataFrame

df_id_age.show(3)


filter() and show() operations

* filter() transformation filters out the rows based on a condition


new_df_age21 = new_df.filter(new_df.Age > 21)

new_df_age21.show(3)

groupby() and count() operations

* groupby() operation can be used to group a variable

test_df_age_group = test_df.groupby("Age")

test_df_age_group.count().show(3)

orderby() Transformations

* orderby() operation sorts the DataFrame based on one o more columns


test_df_age_group.count().orderBy('Age').show()



dropDuplicates()

* dropDuplicates() removes the duplicate rows of a DataFrame


test_df_no_dup = test_df.select('User_ID', 'Gender', 'Age').dropDuplicates()

test_df_no_dup.count()

WithColumnRenamed() renames a column in the DataFrame


test_df_sex = test_df.withColumnRenamed('Gender', 'Sex')

test_df_sex.show(3)


printSchema()

* printSchema() operation prints the types of columns in the DataFrame

test_df.printSchema()

columns actions

* columns operator pirnts the columns of a DataFrame

test_df.columns

describe() actions

* describe() operation compute summary statistics of numerical columns in the DataFrame

"""


"""
Inspecting data in PySpark DataFrame
Inspecting data is very crucial before performing analysis such as plotting, modeling, training etc., In this simple exercise, you'll inspect the data in the people_df DataFrame that you have created in the previous exercise using basic DataFrame operators.

Remember, you already have a SparkSession spark and a DataFrame people_df available in your workspace.

"""

# Print the first 10 observations 
people_df.show(10)

# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and their names
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))



"""
PySpark DataFrame subsetting and cleaning
After data inspection, it is often necessary to clean the data which mainly involves subsetting, renaming the columns, removing duplicated rows etc., PySpark DataFrame API provides several operators to do this. In this exercise, your job is to subset 'name', 'sex' and 'date of birth' columns from people_df DataFrame, remove any duplicate rows from that dataset and count the number of rows before and after duplicates removal step.

Remember, you already have a SparkSession spark and a DataFrame people_df available in your workspace.

"""



# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))




"""
Filtering your DataFrame
In the previous exercise, you have subset the data using select() operator which is mainly used to subset the DataFrame column-wise. What if you want to subset the DataFrame based on a condition (for example, select all rows where the sex is Female). In this exercise, you will filter the rows in the people_df DataFrame in which 'sex' is female and male and create two different datasets. Finally, you'll count the number of rows in each of those datasets.

Remember, you already have a SparkSession spark and a DataFrame people_df available in your workspace.

"""


# Filter people_df to select females 
people_df_female = people_df.filter(people_df.sex == "female")

# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")

# Count the number of rows 
print("There are {} rows in the people_df_female DataFrame and {} rows in the people_df_male DataFrame".format(people_df_female.count(), people_df_male.count()))



"""


Interactin with DataFrames using PySpark SQL



DataFrame API vs SQL queries


* In PySpark you can interact with SparkSQL through DataFrame API and SQL queries

* The DataFrame API provides a programmatic domain-specific language(DSL) for data

* DataFrame transformations and actions are easier to construct programmatically


* SQL queries can be concise and easier to understand and portable

* The operations on DataFrames can also be done using SQL queries


Executing SQL Queries


* The SparkSession sql() method executes SQL query

* sql() method takes a SQL statement as an argument and returns the result as DataFrame

df.createOrReplaceTempView("table1")


df2 = spark.sql("SELECT field1, field2 FROM table1")

df2.collect()

test_df.createOrReplaceTempView("test_table")


query = '''SELECT Product_ID FROM test_table'''


Summarizing and grouping data using SQL queries


test_df.createOrReplaceTempView("test_table")

query = ''' SELECT Age, max(Purchase) FROM test_table GROUP BY Age

spark.sql(query).show(5)


Filtering Columns using SQL queries

test_df.createOrReplaceTempView("test_table")

query = ''' SELECT Age, Purchase, Gender FROM test_table WHERE Purchase > 20000 AND Gender == "F"'''


spark.sql(query).show(5)


"""

"""Running SQL Queries Programmatically
DataFrames can easily be manipulated using SQL queries in PySpark. The sql() function on a SparkSession enables applications to run SQL queries programmatically and returns the result as another DataFrame. In this exercise, you'll create a temporary table of the people_df DataFrame that you created previously, then construct a query to select the names of the people from the temporary table and assign the result to a new DataFrame.

Remember, you already have a SparkSession spark and a DataFrame people_df available in your workspace.
"""

# Create a temporary table "people"
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''

# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)


"""
SQL queries for filtering Table
In the previous exercise, you have run a simple SQL query on a DataFrame. There are more sophisticated queries you can construct to obtain the result that you want and use it for downstream analysis such as data visualization and Machine Learning. In this exercise, we will use the temporary table people that you created previously and filter out the rows where the "sex" is male and female and create two DataFrames.

Remember, you already have a SparkSession spark and a temporary table people available in your workspace.

"""


# Filter the people table to select female sex 
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')

# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * FROM people WHERE sex=="male"')

# Count the number of rows in both DataFrames
print("There are {} rows in the people_female_df and {} rows in the people_male_df DataFrames".format(people_female_df.count(), people_male_df.count()))



"""
Data Visualization in PySpark using DataFrames

What is Data visualization?


* Data visualization is a way of representing your data in graphs or charts

* Open source plotting tools to aid visualization in Python:

	* Matplotlib, Seaborn, Bokeh

Plotting graphs using PySpark DataFrames is done using three methods

* pyspark_dist_explore library
* toPandas()
* HandySpark library

Data Visualization using Pyspark_dist_explore

* Pyspark_dist_explore library provides quick insights into DataFrames

* Currently three functions available - hist(), distplot() and pandas_histogram()


test_df = spark.read.csv("test.csv", header=True, inferSchema=True)

test_df_age = test_df.select('Age')

hist(test_df_age, bins =20, color="red")


Using Pandas for plotting DataFrames

* it's easy to create charts from pandas DraFrames

test_df = spark.read.csv("test.csv", header=True, inferSchema=True)


test_df_sample_pandas = test_df.toPandas()

test_df_sample_pandas.hist('Age')


Pandas DataFrame vs PySpark DataFrame


Pandas DataFrames are in-memory, single-server based structures and operations on PySpark run in parallel

* The result is generated as we apply any operation in Pandas whereas operations in PySpark dataFrame are lazy evaluation

* Pandas DataFrame as mutable and PySpark DataFrames are inmmutable

* Pandas API support more operations than PySpark Dataframe API

HandySpark method of visualization

test_df = spark.read.csv('test.csv', header=True, inferSchema=True)


hdf = test_df.toHandy()

hdf.cols["Age"].hist()
"""

"""
PySpark DataFrame visualization
Graphical representations or visualization of data is imperative for understanding as well as interpreting the data. In this simple data visualization exercise, you'll first print the column names of names_df DataFrame that you created earlier, then convert the names_df to Pandas DataFrame and finally plot the contents as horizontal bar plot with names of the people on the x-axis and their age on the y-axis.

Remember, you already have a SparkSession spark and a DataFrame names_df available in your workspace.

"""

# Check the column names of names_df
print("The column names of names_df are", names_df.columns)

# Convert to Pandas DataFrame  
df_pandas = names_df.toPandas()

# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='Name', y='Age', colormap='winter_r')
plt.show()


"""
Part 1: Create a DataFrame from CSV file
Every 4 years, the soccer fans throughout the world celebrates a festival called “Fifa World Cup” and with that, everything seems to change in many countries. In this 3 part exercise, you'll be doing some exploratory data analysis (EDA) on the "FIFA 2018 World Cup Player" dataset using PySpark SQL which involve DataFrame operations, SQL queries and visualization.

In the first part, you'll load FIFA 2018 World Cup Players dataset (Fifa2018_dataset.csv) which is in CSV format into a PySpark's dataFrame and inspect the data using basic DataFrame operations.

Remember, you already have a SparkSession spark and a variable file_path available in your workspace.
"""




# Load the Dataframe
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the schema of columns
fifa_df.printSchema()

# Show the first 10 observations
fifa_df.show(10)

# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))



"""
Part 2: SQL Queries on DataFrame
The fifa_df DataFrame that we created has additional information about datatypes and names of columns associated with it. This additional information allows PySpark SQL to run SQL queries on DataFrame. SQL queries are concise and easy to run compared to DataFrame operations. But in order to apply SQL queries on DataFrame first, you need to create a temporary view of DataFrame as a table and then apply SQL queries on the created table (Running SQL Queries Programmatically).

In the second part, you'll create a temporary table of fifa_df DataFrame and run SQL queries to extract the 'Age' column of players from Germany.

You already have a SparkContext spark and fifa_df available in your workspace.


"""


# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')

# Construct the "query"
query = '''SELECT "Age" FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()



"""
Part 3: Data visualization
Data visualization is important for exploratory data analysis (EDA). PySpark DataFrame is a perfect for data visualization compared to RDDs because of its inherent structure and schema.

In this third part, you'll create a histogram of the ages of all the players from Germany from the DataFrame that you created in the previous exercise. For this, you'll first convert the PySpark DataFrame into Pandas DataFrame and use matplotlib's plot() function to create a density plot of ages of all players from Germany.

Remember, you already have a SparkSession spark, a temporary table fifa_df_table and a DataFrame fifa_df_germany_age available in your workspace.

"""


# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()








#----------------------------------------------------------------------------------------------------------

"""

Overview of PySpark MLlib


What is PySpark MLlib?
* MLlib is a component of Apache Spark for machine learning

* Varios tools provided by MLlib include:

	* ML Algorithms: collaborative filtering, classification, and clustering

	* Featurization: Feature extraction, transformation,dimensionality reduction, and selection
	
	* Pipelines: tools for constructing, evaluating and tunning ML Pipelines

Why PySpark MLlib?

* Scikit-learn is a popular Python library for data mining and machine learning

* Scikit-learn algorithms only work for small datasets on a single machine

* Spark's MLlib algorithms are designed for paralled processing on a cluster

* Supports languages such as Scala, Java, and R

* Provides a high-level API to build machine learning pipelines



PySpark MLlb Algorithms

* Classificaton (Binary and Multiclass) and Regression: Linear SVMs, logistic regression, decision trees, random forests, gradient-boosted trees, naiv Bayes, linear least squares, Lasso, ridge regression, isotonic  regression

* Collaborative filterng: Alternating least squares(ALS)

* Clustering: K-means, Gaussian mixture, Bisecting k-means and Streamng K-means



The three C'sof machine learning in PySpark MLlib

* Collaborative filtering ( recomender engines): Produce recommendations

* Classification: Identifying to which of a set of categories a new observation

* Clustering: Groups data based on a similar characteristics

PySpark MLlib imports

pyspark.mllib.recommendation

from pyspark.mllib.recommendation import ALS

* pyspark.mllib.classification

from pyspark.mllib.classification import LogisticRegressionWithLBFGS


*pyspark.mllib.clustering


from pyspark.mllib.clustering import KMeans



"""
"""
PySpark MLlib algorithms
Before using any Machine learning algorithms in PySpark shell, you'll have to import the submodules of pyspark.mllib library and then choose the appropriate class that is needed for a specific machine learning task.

In this simple exercise, you'll learn how to import the different submodules of pyspark.mllib along with the classes that are needed for performing Collaborative filtering, Classification and Clustering algorithms.
"""


# Import the library for ALS
from pyspark.mllib.recommendation import ALS

# Import the library for Logistic Regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

# Import the library for Kmeans
from pyspark.mllib.clustering import KMeans

"""
Introduction to Collaborative filtering


What is Collaborative filtering?

* Collaborative filtering is finding users that share common interests

* Collaborative filtering approaches

	* User-User Collaborative filtering: Finds users that are similar to the target user

	* Item-Item Collaborative filtering: Findes and recommends items that are similiar to items with the target user



Rating class in pyspark.mllib.recommendation submodule


* The Rating class is a wrapper around tuple (user, product, and rating)

* Useful for parsing the RDD and creating a tuple of user, product and rating

from pyspark.mllib.recommendation import Rating

r = Rating(user = 1, product = 2, rating = 5.0)

(r[0], r[1], r[2])



Spltting the data using randomSplit()

* Splitting data intro training and testing sets is important for evaluating predictive modeling

* typically a large portion of data is assigned to training compared to testing data


* PysSpark's randomSplit() method randomly splits with the provided weights and returns multiple RDDs


data = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

training, test = data.randomSplit([0.6, 0.4])

training.collect()

test.collect()


Alternating Least Squares(ALS)

* Alternating Least Squares(ALS) algorithm in spark.mllib provides collaborative filtering

* ALS.train(ratings, rank, iterations)


r1 = Rating(1, 1, 1.0)

r2 = Rating(1, 2, 2.0)

r3 = Rating(2, 1, 2.0)

ratings = sc.parallelize([r1, r2, r3 ])

ratings.collect()

model = ALS.train(rating, rank=10, iterations=10)

PredictAll()- Returns RDD of Rating Objects

* The predictAll() method returns a list of predicted ratings for input user and product pair

* The method takes in an RDD without ratings to generate the ratings


unrated_RDD = sc.parellelize([(1, 2), (1, 1)])


predictions = model.predicAll(unrated_RDD)

predictions.collect()


Model Evaluation using MSE

* The MSE is the average value of the square of (actual rating - predicted rating)

rates = ratings.map(lambda x: ((x[0], x[1]), x[2]))


rates.collect()

[((1,1), 1), ((1,2), 2), ((2, 1), 2)]


preds = predictions.map(lambda x: ((x[0], x[1]), x[2]))

preds.collect()

[((1,1), 1.000027), ((1, 2), 1.989999)]





rates_preds = rates.join(preds)

rates_preds.collect()

[((1, 2), (2.0, 1.989000)),((1,1), (1,0000026))]


MSE= rates_preds.map(lambda r: (r[1][0]-r[1][1])**2).mean()
"""


"""
Loading Movie Lens dataset into RDDs
Collaborative filtering is a technique for recommender systems wherein users' ratings and interactions with various products are used to recommend new ones. With the advent of Machine Learning and parallelized processing of data, Recommender systems have become widely popular in recent years, and are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags. In this 3-part exercise, your goal is to develop a simple movie recommendation system using PySpark MLlib using a subset of MovieLens 100k dataset.

In the first part, you'll first load the MovieLens data (ratings.csv) into RDD and from each line in the RDD which is formatted as userId,movieId,rating,timestamp, you'll need to map the MovieLens data to a Ratings object (userID, productID, rating) after removing timestamp column and finally you'll split the RDD into training and test RDDs.

Remember, you have a SparkContext sc available in your workspace. Also file_path variable (which is the path to the ratings.csv file), and ALS class are already available in your workspace.


"""


# Load the data into RDD
data = sc.textFile(file_path)

# Split the RDD 
ratings = data.map(lambda l: l.split(','))

# Transform the ratings RDD 
ratings_final = ratings.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))

# Split the data into training and test
training_data, test_data = ratings_final.randomSplit([0.8, 0.2])





"""
Model training and predictions
After splitting the data into training and test data, in the second part of the exercise, you'll train the ALS algorithm using the training data. PySpark MLlib's ALS algorithm has the following mandatory parameters - rank (the number of latent factors in the model) and iterations (number of iterations to run). After training the ALS model, you can use the model to predict the ratings from the test data. For this, you will provide the user and item columns from the test dataset and finally return the list of 2 rows of predictAll() output.

Remember, you have SparkContext sc, training_data and test_data are already available in your workspace.


"""



# Create the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)

# Drop the ratings column 
testdata_no_rating = test_data.map(lambda p: (p[0], p[1]))

# Predict the model  
predictions = model.predictAll(testdata_no_rating)

# Return the first 2 rows of the RDD
predictions.take(2)



"""
Model evaluation using MSE
After generating the predicted ratings from the test data using ALS model, in this final part of the exercise, you'll prepare the data for calculating Mean Square Error (MSE) of the model. The MSE is the average value of (original rating – predicted rating)**2 for all users and indicates the absolute fit of the model to the data. To do this, first, you'll organize both the ratings and prediction RDDs to make a tuple of ((user, product), rating)), then join the ratings RDD with prediction RDD and finally apply a squared difference function along with mean() to get the MSE.

Remember, you have a SparkContext sc available in your workspace. Also, ratings_final and predictions RDD are already available in your workspace.


"""


# Prepare ratings data
rates = ratings_final.map(lambda r: ((r[0], r[1]), r[2]))

# Prepare predictions data
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))

# Join the ratings data with predictions data
rates_and_preds = rates.join(preds)

# Calculate and print MSE
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))








"""

Classification using PySpark Mllib


* Classification is a supervised machine learning algorithm for sorting the input data into different categories

Binary classification
Multi-class classification




Working with Vector


* PySpark Mllib contains specific data types Vectors and labelledPoint


* Two types of Vectors

* Dense Vector: store all their entries in an array of floating point numbers


* Sparse Vector: store only the nonzero values and their indices

denseVec = Vectors.dense([1.0, 2.0, 3.0])
DenseVector([1.0, 2.0, 3.0]) 

sparseVec= Vectors.sparse(4, {1:10, 3: 5.5})



LabelledPoint() in PySpark Mllib


* A labeledPoint is a wrapper for input features and predicted value


* For Binary classification of logistic Regression, a label is either 0(negative) or 1(positive)


positive = LabeledPoint(1.0, [1.0, 0.0, 3.0])

negative = LabeledPoint(0.0, [2.0, 1.0, 1.0])


print(positive)

print(negative)

HashingTF() in PySpark MLlib



* HashingTF algorithm is used to map feature value to indices in the feature vector


from pyspark.mllib.feature import HashingTF

sentence = "hello hellow world"


words = sentence.split()


tf = HashingTF(10000)

tf.transform(words)




Logistic Regression using  LogisticRegressionWithLBFGS

* logistic Regression using Pyspark MLlib is achieved using LogisticRegressionWithLBFGS


data = [
	
	LabeledPoint(0.0, [0.0, 1.0]),
	LabeledPoint(1.0, [1.0, 0.0]),


]


RDD = sc.parallelize(data)


lrm = LogisticRegressionWithLBFGS.train(RDD)


lrm.predict([1.0, 0.0])

lrm.predict([0.0, 1.0])
"""


"""

Loading spam and non-spam data
Logistic Regression is a popular method to predict a categorical response. Probably one of the most common applications of the logistic regression is the message or email spam classification. In this 3-part exercise, you'll create an email spam classifier with logistic regression using Spark MLlib. Here are the brief steps for creating a spam classifier.

Create an RDD of strings representing email.
Run MLlib’s feature extraction algorithms to convert text into an RDD of vectors.
Call a classification algorithm on the RDD of vectors to return a model object to classify new points.
Evaluate the model on a test dataset using one of MLlib’s evaluation functions.
In the first part of the exercise, you'll load the 'spam' and 'ham' (non-spam) files into RDDs, split the emails into individual words and look at the first element in each of the RDD.

Remember, you have a SparkContext sc available in your workspace. Also file_path_spam variable (which is the path to the 'spam' file) and file_path_non_spam (which is the path to the 'non-spam' file) is already available in your workspace.

"""

# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))

# Print the first element in the split RDD
print("The first element in spam_words is", spam_words.first())
print("The first element in non_spam_words is", non_spam_words.first())



"""
Feature hashing and LabelPoint
After splitting the emails into words, our raw data set of 'spam' and 'non-spam' is currently composed of 1-line messages consisting of spam and non-spam messages. In order to classify these messages, we need to convert text into features.

In the second part of the exercise, you'll first create a HashingTF() instance to map text to vectors of 200 features, then for each message in 'spam' and 'non-spam' files you'll split them into words, and each word is mapped to one feature. These are the features that will be used to decide whether a message is 'spam' or 'non-spam'. Next, you'll create labels for features. For a valid message, the label will be 0 (i.e. the message is not spam) and for a 'spam' message, the label will be 1 (i.e. the message is spam). Finally, you'll combine both the labeled datasets.

Remember, you have a SparkContext sc available in your workspace. Also spam_words and non_spam_words variables are already available in your workspace.

"""


# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)

# Map each word to one feature
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))

# Combine the two datasets
samples = spam_samples.union(non_spam_samples)



"""
Logistic Regression model training
After creating labels and features for the data, we’re ready to build a model that can learn from it (training). But before you train the model, in this final part of the exercise, you'll split the data into training and test, run Logistic Regression model on the training data, and finally check the accuracy of the model trained on training data.

Remember, you have a SparkContext sc available in your workspace, as well as the samples variable.

"""



# Split the data into training and testing
train_samples,test_samples = samples.randomSplit([0.8, 0.2])

# Train the model
model = LogisticRegressionWithLBFGS.train(train_samples)

# Create a prediction label from the test data
predictions = model.predict(test_samples.map(lambda x: x.features))

# Combine original labels with the predicted labels
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)

# Check the accuracy of the model on the test data
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))





"""

Introduction to Clustering



What is Clustering?


* Clustering is the unsupervised learning taks to organize of data into groups

* PySpark MLlib library currently supports the following clustering models

	* K-means
	* Gaussian mixture

	* Power iteration clustering(PIC)

	* Bisecting k-means

	* Streaming k-means
"""


"""
K-means is the most popular clustering method




K-means with Spakr MLlib

RDD = sc.textFile("WineData.csv").\
	map(lambda x: x.split(",")).\
	map.(lambda x: [float(x[0]), float(x[1])])


RDD.take(5)




from pyspark.mllib.clustering import KMeans

model = KMeans.train(RDD, k=2, maxIterations = 10)

model.clusterCenters

Evaluating the K-means Models

from math import sqrt

def error(point):
	center= model.centers[model.predict(point)]

	return sqrt(sum([x**2  for x in (point-center)]))


#Within set sum of squared error
WSSSE = RDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)

print("Within Set Sum of Squared Error = " + str(WSSSE))


Visualizing clusters

wine_data_df = spark.createDataFrame(RDD, schema=["col1","col2"])

wine_data_df_pandas = wine_data_df.toPandas()


cluster_centers_pandas = pd.DataFrame(model.clusterCenters, columns=["col1", "col2"])

cluster_centers_pandas.head()



plt.scatter(wine_data_df_pandas["col1"], wine_data_df_pandas["col2"]);

plt.scatter(cluster_centers_pandas["col1", cluster_centers_pandas["col2"], color="red", marker="x"])


"""

"""
Loading and parsing the 5000 points data
Clustering is the unsupervised learning task that involves grouping objects into clusters of high similarity. Unlike the supervised tasks, where data is labeled, clustering can be used to make sense of unlabeled data. PySpark MLlib includes the popular K-means algorithm for clustering. In this 3 part exercise, you'll find out how many clusters are there in a dataset containing 5000 rows and 2 columns. For this you'll first load the data into an RDD, parse the RDD based on the delimiter, run the KMeans model, evaluate the model and finally visualize the clusters.

In the first part, you'll load the data into RDD, parse the RDD based on the delimiter and convert the string type of the data to an integer.

Remember, you have a SparkContext sc available in your workspace. Also file_path variable (which is the path to the 5000_points.txt file) is already available in your workspace.

"""
# Load the dataset into an RDD
clusterRDD = sc.textFile(file_path)

# Split the RDD based on tab
rdd_split = clusterRDD.map(lambda x: x.split("\t"))

# Transform the split RDD by creating a list of integers
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])

# Count the number of rows in RDD 
print("There are {} rows in the rdd_split_int dataset".format(rdd_split_int.count()))



"""
K-means training
Now that the RDD is ready for training, in this 2nd part, you'll test with k's from 13 to 16 (to save computation time) and use the elbow method to chose the correct k. The idea of the elbow method is to run K-means clustering on the dataset for different values of k, calculate Within Set Sum of Squared Error (WSSSE) and select the best k based on the sudden drop in WSSSE. Next, you'll retrain the model with the best k and finally, get the centroids (cluster centers).

Remember, you already have a SparkContext sc and rdd_split_int RDD available in your workspace.


"""


# Train the model with clusters from 13 to 16 and compute WSSSE
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k
model = KMeans.train(rdd_split_int, k=15, seed=1)

# Get cluster centers
cluster_centers = model.clusterCenters





"""
Visualizing clusters
You just trained the k-means model with an optimum k value (k=15) and generated cluster centers (centroids). In this final exercise, you will visualize the clusters and the centroids by overlaying them. This will indicate how well the clustering worked (ideally, the clusters should be distinct from each other and centroids should be at the center of their respective clusters).

To achieve this, you will first convert the rdd_split_int RDD into a Spark DataFrame, and then into Pandas DataFrame which can be used for plotting. Similarly, you will convert cluster_centers into a Pandas DataFrame. Once both the DataFrames are created, you will create scatter plots using Matplotlib.

The SparkContext sc as well as the variables rdd_split_int and cluster_centers are available in your workspace.

"""

# Convert rdd_split_int RDD into Spark DataFrame and then to Pandas DataFrame
rdd_split_int_df_pandas = spark.createDataFrame(rdd_split_int, schema=["col1","col2"]).toPandas()

# Convert cluster_centers to a pandas DataFrame
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])


""""

plt.scatter(wine_data_df_pandas["col1"], wine_data_df_pandas["col2"]);

plt.scatter(cluster_centers_pandas["col1", cluster_centers_pandas["col2"], color="red", marker="x"])


"""
# Create an overlaid scatter plot of clusters and centroids
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()