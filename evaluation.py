import os

import pandas as pd
import sqlite3
import torch

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    EvaluationResult
)

# seed the model
torch.manual_seed(42)

model_cache = "path to models"
os.environ['HF_HOME'] = model_cache

# read in data from tracking db
conn = sqlite3.connect("path to tracking database")
c = conn.cursor()
c.execute('SELECT * FROM llama_results') 
rows = c.fetchall()

# construct dataframe from rows
column_names = [description[0] for description in c.description]
eval_df = pd.DataFrame(columns=column_names)
for row in rows:
    df_row = pd.DataFrame(row).T.set_axis(column_names, axis=1)
    eval_df = pd.concat([eval_df, df_row])

# initialize LLM judges
judges = {}

model_name="model name"
tokenizer_name="same as model name"
evaluator_llm = HuggingFaceLLM(model_name=model_name, 
                               tokenizer_name=tokenizer_name,
                               model_kwargs={"cache_dir": rsma_cache},
                               tokenizer_kwargs={"cache_dir": rsma_cache},
                               generate_kwargs={"temperature":0.1, "do_sample":True},
                               device_map="auto"
)

judges["answer_relevancy"] = AnswerRelevancyEvaluator(
    llm=evaluator_llm,
)

judges["context_relevancy"] = ContextRelevancyEvaluator(
    llm=evaluator_llm,
)

judges["faithfulness"] = FaithfulnessEvaluator(
    llm=evaluator_llm,
)

# example evaluation for first response
eval_row = eval_df.iloc[-1]

# construct response object
response = Response(eval_row["response"], eval(eval_row["source_nodes"]))

faithfulness_result = judges["faithfulness"].evaluate_response(response=response)

context_rel_result = judges["context_relevancy"].evaluate_response(query=eval_row["hyde_prompt"],response=response)

answer_rel_result = judges["answer_relevancy"].evaluate_response(query=eval_row["hyde_prompt"],response=response)

# just silly stuff
for node in eval(eval_row["retrieved_nodes"]):
    #print(node.text)
    print(node.metadata["window"])
    print()

for node in eval(eval_row["source_nodes"]):
    print(node.text)
    #print(node.metadata["window"])
    print()