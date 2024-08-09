import argparse
import os
import json

from llama_index.core import PromptTemplate, Settings
from llama_index.core.query_pipeline import QueryPipeline

import util

# this helps when GPU is out of memory?
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# current crop of local models we are using
model_name="LLM"
tokenizer_name="tokenizer - same as LLM usually"
embed_model_name = "embedding model"
reranker_model_name = "reranker model"

def execute_hyde(user_prompt):

    # generate prompt to rewrite the user prompt
    prompt_str_1 = "Please edit the following prompt to be a more precisely phrased, detailed question while preserving the original meaning: {user_prompt}"
    prompt_tmpl_1 = PromptTemplate(prompt_str_1)

    # generate prompt for hallucinating answer
    prompt_str2 = (
        "Please write a passage to answer the question\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "\n"
        "{query_str}\n"
        "\n"
        "\n"
        'Passage:"""\n'
    )
    prompt_tmpl2 = PromptTemplate(prompt_str2)

    # first query pipeline is for HyDE
    # generates better question and sample answer for lookup
    hyde_pipeline = QueryPipeline(
        chain=[prompt_tmpl_1, Settings.llm, prompt_tmpl2, Settings.llm], verbose=True
    )

    hyde_results, hyde_intermediates = hyde_pipeline.run_with_intermediates(user_prompt)

    # get the prompt rewrite and the hypothetical answer
    hyde_prompt = hyde_intermediates[list(hyde_intermediates.keys())[1]].outputs["output"].text
    hyde_answer = hyde_results.text

    return {"hyde_prompt": hyde_prompt,
            "hyde_answer": hyde_answer}


    
def execute_rag(pipeline_kwargs, pipeline_objs):

    # parse the pipeline args
    user_prompt = pipeline_kwargs["user_prompt"]
    rerank = pipeline_kwargs["rerank"]
    hyde = pipeline_kwargs["hyde"]
    sentence_window = pipeline_kwargs["sentence_window"]

    # execute HyDE
    if hyde:
        hyde_results = execute_hyde(user_prompt=user_prompt)

        # if doing HyDE we use the hallucinated answer for retrieval
        # and ask the LLM to answer the rewritten question
        retrieval_str = hyde_results["hyde_answer"]
        query_str = hyde_results["hyde_prompt"]

    else:  
        retrieval_str = user_prompt
        query_str = user_prompt

    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # initialize custom query engine
    query_engine = util.RAGQueryEngine(
        retriever=pipeline_objs["retriever"], 
        rerank=rerank,
        reranker=pipeline_objs["reranker"],
        sentence_window=sentence_window,
        sentence_window_processor=pipeline_objs["sentence_window_processor"],
        llm=Settings.llm, 
        qa_prompt=qa_prompt, 
        retrieval_str=retrieval_str
    )

    response, retrieved_nodes, source_nodes, final_prompt = query_engine.query(query_str)

    return {"response": response,
            "retrieved_nodes": retrieved_nodes,
            "source_nodes": source_nodes,
            "final_prompt": final_prompt,
            "query_str": query_str,
            "retrieval_str": retrieval_str}

def main():

    # parse args
    args_dict = util.parse_args()

    # load embedding model into Settings
    util.load_embedding_model(embed_model_name=embed_model_name)

    # load LLM into Settings
    llm_kwargs = args_dict["llm_kwargs"]
    util.load_llm(model_name=model_name,
                  tokenizer_name=tokenizer_name,
                  llm_kwargs=llm_kwargs)

    # load retriever
    retriever_kwargs = args_dict['retriever_kwargs']
    retriever = util.get_retriever(retriever_kwargs)

    # load reranker
    reranker_kwargs = args_dict["reranker_kwargs"]
    reranker = util.get_reranker(reranker_model_name=reranker_model_name,
                                 reranker_kwargs=reranker_kwargs)
    
    # load sentence window replacement postprocessor
    sentence_window_processor = util.get_sentence_window_processor()

    # pass kwargs and objects we have loaded in to the RAG
    pipeline_kwargs = args_dict["pipeline_kwargs"]
    pipeline_objs = {"retriever": retriever,
                     "reranker": reranker,
                     "sentence_window_processor": sentence_window_processor}
    
    results = execute_rag(pipeline_kwargs=pipeline_kwargs,
                          pipeline_objs=pipeline_objs)

    print(f"User prompt: {pipeline_kwargs['user_prompt']}")
    print()

    if pipeline_kwargs['hyde']:
        print(f"LLM-rewritten prompt: {results['query_str']}")
        print()
        print(f"LLM-hallucinated answer : {results['retrieval_str']}")
        print()

    print(f"Sources retrieved and reranked:")
    for i, source in enumerate(results['source_nodes']):
        print(f"{i+1}. Date: {source.metadata['date']}, Report Section: {source.metadata['report_section']}, Score: {source.score}")
        print(source.text)
        print()

    #print(f"Final prompt: \n{results['final_prompt']}")

    print("Results:")
    print(f"{results['response']}")

    db_args = {"user_prompt": pipeline_kwargs["user_prompt"],
               "hyde_prompt": results["query_str"] if pipeline_kwargs["hyde"] else "",
               "hyde_answer": results["retrieval_str"] if pipeline_kwargs["hyde"] else "",
               "final_prompt": results["final_prompt"],
               "response": results["response"],
               "retrieved_nodes": results["retrieved_nodes"],
               "source_nodes": results["source_nodes"],
               "temperature": llm_kwargs["temperature"],
               "similarity_top_k": retriever_kwargs["similarity_top_k"],
               "collection": retriever_kwargs["collection"],
               "metadata_filter": retriever_kwargs["metadata_dict"],
               "rerank": 1*pipeline_kwargs["rerank"],
               "top_n": reranker_kwargs["top_n"],
               "sentence_window": 1*pipeline_kwargs["sentence_window"]
               }
    
    util.sqlite_upload(db_args=db_args)

if __name__ == '__main__':
    main()
