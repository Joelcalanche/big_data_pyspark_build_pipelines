# Unit test for PySpark
"""

In this lesson, we will rewrite the transformations in this piece of code from our pipeline to allow testing.

prices_with_ratings = spark.read.csv(...) # extract

exchange_rates = spark.read.csv() # extract


unit_prices_with_ratings = (prices_with_ratings
							.join(...)  # transform
							.withColumn(...) # transform)


para permitir test debemos aislar estas partes


# Extract the data

df = spark.read.csv(path_to_file)

hacer test de esta manera no es lo mas eficiente
primero podemos crear un dummy dataset y ver como va

ejemplo de dummy data set


from pyspark.sql import Row

purchase = Row("price",
               "quantity",
               "product")
record = purchase(12.99, 1, "cake")
# esto separece a crear en pandas a partir de una serie
df = spark.createDateFrame((record,))

esto nos permite ver con claridad cual es la entrada

y la data es cerrada en donde esta siendo usada

siempre escribe test con dummy dataset

las desventajas de no hacerlo son
- dependemos de las entradas y salidas
- no se sabe que tan grande sera la data

- no sabemos que data viene


unit_prices_with_ratings = (prices_with_ratings
							.join(exchange_rates, ["currency", "date"])
							.withColumn("unit_price_in_euro",
										col("price") / col("quantity")
										* col("exchange_rate_to_euro")))


podemos separar estos pasos 

write out each step to its own function

def link_with_exchange_rates(prices, rates):
		return prices.join(rates, ["currency", "date"])

def calculate_unit_price_in_euro(df):
		return df.withColumn(
			"unit_price_in_euro",
			col("price")/ col("quantitiy") * col("exchange_rate_to_euro"))



luego de haber separado en funciones, las aplicamos en secuencia

# lo primero que debe ejecutarse es la funcion mas interna joel 
unit_prices_with_ratings = (
	calculate_unit_price_in_euro(
		link_with_exchange_rates(prices, exchange_rates)
	)
)

cada transformacion por separado debe ser probada


pasos:
1) creamos  un dataframe en memoria
2)  debemos pasar este dataframe a la funcion que queremos probar
3) nuestra expectation is that th "unit_price_in_euro" field is added and that the math is correct


def test_calculate_unit_price_in_euro():
		record = dict(price=10,
		 			  quantity=5,
		 			  exchange_rate_to_euro=2.)

		df = spark.createDataFrame([Row(**record)])
		result = calculate_unit_price_in_euro(df)

		expected_record = Row(**record, unit_price_in_euro=4.)
		expected = spark.createDataFrame([expected_record])
		# con esta funcion conseguimos probar nuestras aserciones
		assertDataFrameEqual(result, expected)


lo importanta de esta leccion es 

1. Interacting with external data sources is costly

2. Creating in-memory DataFrames makes testing easier

* the data is a plain sight.
* focus in on just small number of examples

and your code be refactored so that you create small, reusable and well-named functions which are also easier to test

3. Creating small and well-named functions leads to more reusability and easier testing.



Continuos testing

para correr multiples pruebas podemos usar varios modulos
en este caso tenemos

in stdlib
unittest
doctest

3rd party
pytest
nose

core task: assert or raise

Examples:
assert computed == expected

with pytest.raises(ValueError): #pytest specific

	la caracteristica principal de estos frameworks es que me generan reportes luego de correr
	y me dicen que test pasaron y cuales no, ademas de alguna informacion para hacer debugging


	una nota importante 2 segundos para un solo test es demasiado



	Spark and othr distributed computing frameworks add overhead in tests.
	--- Spark increases time to run unit tests


	Automating test


	Automation is one of the prime objectives of a data engineer

	Running unit tests manually is tedious and error-prone, as you might forget to run it after a series of changes to your code.

	solution:
	* Automation


	if you're working professionally with code, you're using some sort of version control system, like git

	In many version control systems you can configure certain scripts to be run when you change code

	* Git--> configure hooks

	* Configure CI/CD pipeline to run test automatically


	CI/CD stands for Continuos Integration and Continuos Delivery

	CI Continuos Integration:

	* get code changes integrated with the master branch regularly

	esto solo permite agregar los cambios al programa principal si cualquier cambio no rompe nada, estas roturas son las que los test ayudan a detectar


	CD that all artifacts should always be in deployable state at any time without any problem

	* Create "artifact"(deliverables like documentation, but also programs) that can be deployed into production without breaking things.

	* static code checks like compliancy with PEP8

	circleci is a services that runs test automatically for you

	muchas de estas librerias buscan un specific file in your code repository, para este en especifico es config.yml

	CircleCI looks for .circleci/config.yml, which should be at the root of your code repository

	* UN archivo yml, es un super  JSON pero con diferents cosas y varias caracteristicas

	example:

	jobs:
		test:
			docker:
				- image: circleci/python:3.6.4
			steps:
				- checkout
				- run: pip install -r requirements.txt
				- run: pytest .

cada job es una coleccion de pasos  que deben ejecutarse en algun environment, los steps deben hacerse secuencialmente


en este caso por ejemplo le deciemos a CircleCI que busque el code from the code repository

1) checkout code(within repository)
2) install test & build requirements
3). run tests
# si todos los tests fueron bien
4) package/build the software artefacts 

you could package the application and deploy it on a server or store it for later use
2)
	CD Continuos Delivery:



ejemplo de yml file  donde se define un conjunto de  tareas:


version: 2
jobs:
  build:
    working_directory: ~/data_scientists/optimal_diapers/
    docker:
      - image: gcr.io/my-companys-container-registry-on-google-cloud-123456/python:3.6.4
    steps:
      - checkout
      - run:
          command: |
            sudo pip install pipenv
            pipenv install
      - run:
          command: |
            pipenv run flake8 .
            pipenv run pytest .
      - store_test_results:
          path: test-results
      - store_artifacts:
          path: test-results
          destination: tr1














from datetime import date
from pyspark.sql import Row

Record = Row("country", "utm_campaign", "airtime_in_minutes", "start_date", "end_date")

# Create a tuple of records
data = (
  Record("USA", "DiapersFirst", 28, date(2017, 1, 20), date(2017, 1, 27)),
  Record("Germany", "WindelKind", 31, date(2017, 1, 25), None),
  Record("India", "CloseToCloth", 32, date(2017, 1, 25), date(2017, 2, 2))
)

# Create a DataFrame from these records
frame = spark.createDataFrame(data)
frame.show()






"""