import singer


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

# data that belong together, should be sent to the same stream

# key_properties=["id"] define una lista para crear la primary key que for records(rows) que se envian por este stream


singer.write_schema(schema=json_schema, stream_name='DC_employees', key_properties=["id"])