import argparse
import os

from copy import deepcopy
import chromadb
import json
import sqlite3
import torch
from llama_index.core import PromptTemplate, QueryBundle, Settings, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore

# this is where we get the LLMs!
model_cache = "path to local models"
os.environ['HF_HOME'] = model_cache

# created new versions of chroma db with llama-index wrappers
DB_DIR = "path to chroma databases"

# for keeping track of results
TRACKING_DB_DIR = "path to tracking databases"

# seed the model
torch.manual_seed(42)

def parse_args():
    """ Parses command line args for the rest of the functions

        Returns:
            args_dict (dict)
           
    """
    parser = argparse.ArgumentParser(description='RAG experimentation with different parameters')

    # add command line options
    parser.add_argument('-t', '--temperature', type=float, default=-1, help='Temperature of LLM, if not set then greedy search is used')

    parser.add_argument('-p', '--user_prompt', type=str, default="What might be the downside, knock-on effects of further deterioration in commercial real estate?", help='Question for the LLM')
    parser.add_argument("--hyde", action='store_true', help='Apply HyDE for query transformation?')

    parser.add_argument('-k', "--similarity_top_k", type=int, default=10, help='Number of sources to retrieve, default is 10')
    parser.add_argument('-c', "--collection", type=str, default="us_fed_fsr", help='Collection to query in Chroma')
    parser.add_argument('-y', "--year", type=str, help='Metadata filter: year documents were published. If providing multiple, separate by commas with no spaces.')
    parser.add_argument('-m', "--month", type=str, help='Metadata filter: month documents were published. If providing multiple, separate by commas with no spaces.')
    parser.add_argument('-s', "--report_section", type=str, help='Metadata filter: section of document to pull from. If providing multiple, separate by commas with no spaces.')

    parser.add_argument("--rerank", action='store_true', help='Apply re-ranking?')
    parser.add_argument('-n', "--top_n", type=int, default=5, help='How many re-ranked results to return?')
    parser.add_argument("--sentence_window", action='store_true', help='Use sentence window retrieval?')

    # parse the command line arguments
    args = parser.parse_args()

    # store args
    retriever_kwargs = {"similarity_top_k": args.similarity_top_k,
                        "collection": args.collection,
                        "metadata_dict": {"year": args.year,
                                          "month": args.month,
                                          "report_section": args.report_section}}
    
    llm_kwargs = {"temperature": args.temperature,
                  "do_sample": (args.temperature > 0)}
    
    reranker_kwargs = {"top_n": args.top_n}
    
    pipeline_kwargs = {"user_prompt": args.user_prompt,
                       "hyde": args.hyde,
                       "rerank": args.rerank,
                       "sentence_window": args.sentence_window}

    args_dict = {"retriever_kwargs": retriever_kwargs,
                 "llm_kwargs": llm_kwargs,
                 "reranker_kwargs": reranker_kwargs,
                 "pipeline_kwargs": pipeline_kwargs}

    return args_dict

def load_embedding_model(embed_model_name):
    # custom embedding model!
    # https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/

    # intitialize instructor embedding model (same as used for vector DB setup)
    embed_model = InstructorEmbedding(text_instruction="Represent the financial sentence query for retrieval: ",
                                      model_name=embed_model_name,
                                      device="cuda")

    # update settings so this one is always used
    Settings.embed_model = embed_model

    # not returning for now but maybe we will at some point if we are using different ones

def generate_completion_to_prompt(system_prompt):
    # transform a string into input zephyr-specific input using the system prompt we want
    def completion_to_prompt(completion):
        return f"<|system|>\n</s> {system_prompt} \n<|user|>\n{completion}</s>\n<|assistant|>\n"
    return completion_to_prompt


def messages_to_prompt(messages):
    # transform a list of chat messages into zephyr-specific input
    # not currently used for anything
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt

def load_llm(model_name, tokenizer_name, llm_kwargs):
    # custom LLM!
    # https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/

    # I guess these should be parameters too?
    context_window = 3900
    max_new_tokens = 512

    # generate completion to prompt function given a system prompt
    system_prompt = "You are an economist with a specialization in assessing financial stability risks to the U.S. economy. You are tasked with assessing the financial system using your expertise in both economics and finance. You answer in precise language and do not use information outside of the provided text."
    completion_to_prompt = generate_completion_to_prompt(system_prompt)

    Settings.llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs=llm_kwargs,
        model_kwargs={"cache_dir": model_cache},
        tokenizer_kwargs={"cache_dir": model_cache},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="auto",
    )

    # not returning anything for now, maybe one day we will have multiple LLMs!


def get_retriever(retriever_kwargs):
    similarity_top_k = retriever_kwargs["similarity_top_k"]
    metadata_dict = retriever_kwargs["metadata_dict"]

    # construct metadata filter
    year = metadata_dict["year"]
    month = metadata_dict["month"]
    report_section = metadata_dict["report_section"]

    llama_filters = []
    # I had to go into the source code and add support for IN for chroma
    # so somebody running this without my venv will get an error here
    # https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.pyhttps://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py
    # the _transform_chroma_filter_operator is what I messed with here... chroma supports $in, not sure why it's not in llama-index
    if year is not None:
        llama_filters.append(MetadataFilter(key="year", value=[int(y) for y in year.split(",")], operator=FilterOperator.IN))
    if month is not None:
        llama_filters.append(MetadataFilter(key="month", value=[int(m) for m in month.split(",")], operator=FilterOperator.IN))
    if report_section is not None:
        llama_filters.append(MetadataFilter(key="report_section", value=report_section.split(","), operator=FilterOperator.IN))

    # vector index setup
    collection_name = retriever_kwargs["collection"]

    db = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db.get_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    # retriever!
    # we can improve this using a custom retriever to combine multiple types
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k, 
        filters=MetadataFilters(filters=llama_filters, condition='and'),
    )

    return retriever

def get_reranker(reranker_model_name, reranker_kwargs):

    # reranker!
    # https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/SentenceTransformerRerank/
    
    top_n = reranker_kwargs["top_n"]
    reranker = SentenceTransformerRerank(
        model=reranker_model_name, top_n=top_n
    )

    return reranker

def get_sentence_window_processor():
    # adds surrounding context stored in "window" parameter in metadata
    # sentence window retrieval!
    # https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/
    sentence_window_processor = MetadataReplacementPostProcessor(target_metadata_key="window")

    return sentence_window_processor

class RAGQueryEngine(CustomQueryEngine):
    # custom query engine!
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine/
    retriever: BaseRetriever
    rerank: bool
    reranker: SentenceTransformerRerank
    sentence_window: bool
    sentence_window_processor: MetadataReplacementPostProcessor
    llm: HuggingFaceLLM
    qa_prompt: PromptTemplate
    retrieval_str: str


    def custom_query(self, query_str: str):
        # retrieve from db based on retrieval_str (hallucinated answer)
        source_nodes = self.retriever.retrieve(self.retrieval_str)
        retrieved_nodes = deepcopy(source_nodes)

        if self.sentence_window:
            # replace with sentence windows
            source_nodes = self.sentence_window_processor.postprocess_nodes(source_nodes)
        
        if self.rerank:
            # rerank based on similarity to prompt
            # either rewritten prompt (if doing HYDE)
            # or base user prompt
            source_nodes = self.reranker.postprocess_nodes(source_nodes, QueryBundle(query_str))

        context_str = "\n\n".join([n.node.get_content() for n in source_nodes])
        final_prompt =  self.qa_prompt.format(context_str=context_str, query_str=query_str)
        
        response = self.llm.complete(final_prompt)

        return response, retrieved_nodes, source_nodes, final_prompt
    
def sqlite_upload(db_args):

    conn = sqlite3.connect(TRACKING_DB_DIR)
    c = conn.cursor()

    c.execute("""INSERT INTO llama_results (
                    user_prompt, 
                    hyde_prompt, 
                    hyde_response, 
                    final_prompt,
                    response, 
                    retrieved_nodes, 
                    source_nodes, 
                    temperature, 
                    similarity_top_k, 
                    collection, 
                    metadata_filter, 
                    rerank, 
                    top_n, 
                    sentence_window, 
                    user)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (db_args["user_prompt"],
               db_args["hyde_prompt"],
               db_args["hyde_answer"],
               db_args["final_prompt"],
               str(db_args["response"]),
               str(db_args["retrieved_nodes"]),
               str(db_args["source_nodes"]),
               db_args["temperature"],
               db_args["similarity_top_k"],
               db_args["collection"],
               json.dumps(db_args["metadata_filter"]),
               db_args["rerank"],
               db_args["top_n"],
               db_args["sentence_window"],
               os.getenv("USER")))
    
    conn.commit()
    conn.close()
