import os
import datetime as dt
import pandas as pd

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from nltk import sent_tokenize

# embedding model
model_name = "embedding model"

# input and output dir
DATA_DIR = "scraped documents"
DB_DIR = "chroma dbs"

# output collection name
collection_name="collection name"

df = pd.DataFrame()
for fname in os.listdir(DATA_DIR):
    df = pd.concat([df, pd.read_csv(f"{DATA_DIR}{fname}")])

df = df.rename(columns={"body":"paragraph"})

# it's easier for now to have year/month as their own fields
df['year'] = pd.to_datetime(df['date']).apply(lambda x: x.year)
df['month'] = pd.to_datetime(df['date']).apply(lambda x: x.month)

# nltk sentence tokenizer
df['embed_chunk'] = df['paragraph'].apply(sent_tokenize)

# split so it's one row per sentence
df = df.explode("embed_chunk").reset_index().drop("index", axis=1)

# create windows
df['prev_sentence'] = df['embed_chunk'].shift()
df['next_sentence'] = df['embed_chunk'].shift(-1)

df = df.fillna("")
df['window'] = df['prev_sentence'] + " " + df['embed_chunk'] + " " + df['next_sentence']

df = df[["country", 
         "date", 
         "year",
         "month",
         "institution", 
         "report_name", 
         "report_section", 
         "paragraph_id", 
         "paragraph",
         "window",
         "embed_chunk"]]

print("Initializing embeddings!")
embeddings = InstructorEmbedding(
    text_instruction="Represent the financial sentence query for retrieval: ",
    model_name=model_name,
    device="cuda"
)

nodes = [TextNode(text=f'{d[10]}', 
                  metadata={"country": d[0], 
                            "date": d[1], 
                            "year":d[2], 
                            "month":d[3], 
                            "institution":d[4], 
                            "report_name":d[5],
                            "report_section": d[6],
                            "paragraph_id": d[7],
                            "paragraph": d[8],
                            "window":d[9]}) for d in zip(df['country'].tolist(),
                                                         df['date'].tolist(),
                                                         df['year'].tolist(),
                                                         df['month'].tolist(),
                                                         df['institution'].tolist(),
                                                         df['report_name'].tolist(),
                                                         df['report_section'].tolist(),
                                                         df['paragraph_id'].to_list(),
                                                         df['paragraph'].to_list(),
                                                         df['window'].to_list(),
                                                         df['embed_chunk'].to_list())]

# initialize client and add documents
client = chromadb.PersistentClient(path=DB_DIR)
chroma_collection = client.get_or_create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embeddings)

# persist
#index.storage_context.persist(DB_DIR)



