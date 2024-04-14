from pymilvus import DataType, MilvusClient

client = MilvusClient(uri="http://192.168.0.239:19530")

schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)

schema.add_field(field_name="object_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="output", datatype=DataType.FLOAT_VECTOR, dim=2048)
schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="img_name", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="group", datatype=DataType.VARCHAR, max_length=10000)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="output",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 10},
)
client.create_collection(
    collection_name="example",
    dimension=5,
    schema=schema,
    index_params=index_params,
)

res = client.get_load_state(collection_name="example")
