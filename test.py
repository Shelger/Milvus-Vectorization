import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

connections.connect("default", host="localhost", port="19530")
has = utility.has_collection("chapter_collection")
print(f"Does collection hello_milvus exist in Milvus: {has}")

if not has:
# if has:
#     existing_collection = Collection(name="chapter_collection")
#     existing_collection.drop()
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields=fields, description="Chapter Embeddings")
    collection_name = "chapter_collection"
    chapter_collection = Collection(name=collection_name, schema=schema)

    with open('pap.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    vectors = []
    batch_size = 10
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        vectors = []

        for line in batch_lines:
            inputs = tokenizer(line, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
            vectors.append(vector)
        
        data_to_insert = [vectors]
        chapter_collection.insert(data_to_insert)
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        chapter_collection.create_index("vector", index)

chapter_collection = Collection(name="chapter_collection")
# Get the number of entities
num_entities = chapter_collection.num_entities
print(f"Number of entities in {chapter_collection}: {num_entities}")

chapter_collection.load()
input_string = '"Why, my dear, you must know, Mrs. Long says that Netherfield '
inputs = tokenizer(input_string, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
input_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

search_results = chapter_collection.search([input_vector], "vector", {"nprobe": 10}, 5)
print(f"Number of hits: {len(search_results)}")
if search_results:
    print(search_results[0])


