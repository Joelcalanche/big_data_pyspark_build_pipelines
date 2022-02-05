#singer


columns = ('id', 'name', 'age', 'has_children')

users ={(1, 'Adrian', 32, False),
		(2, 'Ruanne', 28, False),
		(3, 'Hillary', 29, True)
}


json_schema = {
	"properties": {"age": {"maximum":130
						   "minimum":1,
						   "type": "integer"},
				   "has_children": {"type":"boolean"},
				   "id": {"type":"integer"},
				   "name": {"type":"string"}},
	"$id": "http://yourdomain.com/schemas/my_user_schema.json",
	"$schema": "http://json-schema.org/draft-07/schema#"
}




# en json el metodo dumps transforma en cadena, mientras que dump me permit escribe un json en un file




# Complete the JSON schema
schema = {'properties': {
    'brand': {'type': 'string'},
    'model': {'type': 'string'},
    'price': {'type': 'number'},
    'currency': {'type': 'string'},
    'quantity': {'type': 'integer', 'minimum': 1},
    'date': {'type': 'string', 'format': 'date'}, 
    'countrycode': {'type': 'string', 'pattern': "^[A-Z]{2}$"},
    'store_name': {'type': 'string'}}}

# Write the schema
singer.write_schema(stream_name='products', schema=schema, key_properties=[])



# Streaming record messages
# el stream name debe coincidir con el dato en el schema
singer.write_record(stream_name="DC_employees",
	record=dict(zip(columns, users.pop())))


fixed_dict = {"type": "RECORD", "stream": "DC_employees"}


record_msg = {**fixed_dict, "record":dict(zip(columns, users.pop()))}

print(json.dumps(record_msg))


import singer
singer.write_schema(stram_name="foo", schema=...)
# record 1, records varios
singer.write_records(stream_name="foo", records=...)



#Ingestion pipeline: Pipe the tap's output into a Singer target. using the | symbol(linux & MacOS)


# python my_tap.py | target-csv
# its goal is to create CSV files from json lines


# con esto conseguimon modularity

#each tap or target is designed to do one thing very well.

#python my_tap.py | target-csv --config userconfig.cfg


#By working with a standardized intermediate format you could easily swap out the "target-csv", for "target-google-sheets" or "target-postgresql" which write their output to whole different systems
# podemos seleccionar que tap y que target por separado

#Singer's STATE messages

singer.write_state(value={"max-last-updated-on": some_variable}) 
# el value es un objeto json

# Run this tap-mydelta on 2019-06-14 at 12:00:00.000 + 02:00 (2nd row wasn't yet present then):



# usando apis

import requests
endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Create the web API’s URL
authenticated_endpoint = "{}/{}".format(endpoint,api_key)

# Get the web API’s reply to the endpoint
api_response = requests.get(authenticated_endpoint).json()
pprint.pprint(api_response)


endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Create the web API’s URL
authenticated_endpoint = "{}/{}".format(endpoint, api_key)

# Get the web API’s reply to the endpoint
api_response = requests.get(authenticated_endpoint).json()
pprint.pprint(api_response)

# Create the API’s endpoint for the shops
shops_endpoint = "{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "shops")
shops = requests.get(shops_endpoint).json()
print(shops)

# Create the API’s endpoint for items of the shop starting with a "D"
items_of_specific_shop_URL = "{}/{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "items", "DM")
products_of_shop = requests.get(items_of_specific_shop_URL).json()
pprint.pprint(products_of_shop)












# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

for shop in requests.get(SHOPS_URL).json()["shops"]:
    # Write all of the records that you retrieve from the API
    singer.write_records(
      stream_name="products", # Use the same stream name that you used in the schema
      records=({**item, "store_name": shop}
               for item in retrieve_products(shop))
    )