from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from argparse import ArgumentParser
from tqdm import tqdm

import os
from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
    ReActAgentWorker,
)

import LLAMA
from pathlib import Path
import json
import dotenv
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool


import requests
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.tools import BaseTool


import nest_asyncio

nest_asyncio.apply()

# org-sTxtc2xK72iD69fFbS3s19GK
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter


from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

from datetime import datetime
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI

def build_agents(documents: list):
   
    Settings.llm = OpenAI(temperature=0.2, model="gpt-4", system_prompt="You are a helpful assistant designed to answer a user's query. Use any of the tools available to you but DO NOT rely on prior knowledge!")
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", max_length=512)


    agents = {}
    query_engines = {}

    all_tools = []

    full_nodes = LLAMA.split_document("full.pdf")
    # full_index = VectorStoreIndex(full_nodes, embed_model=embed_model)

    if not os.path.exists(f"./data/full.pdf"):

        full_index = VectorStoreIndex(full_nodes, embed_model=embed_model)
        full_index.storage_context.persist(
            persist_dir=f"./data/full.pdf"
            )
    else:
        full_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/full.pdf"),
            )

    full_summary_index = SummaryIndex(full_nodes)

    full_query_engine = full_index.as_query_engine(llm=Settings.llm)
    full_summary_engine = full_summary_index.as_query_engine(llm=Settings.llm)

    textbook_tools = [
        QueryEngineTool(
                query_engine=full_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        "the science of computing."
                    ),
                ),
            ),


        QueryEngineTool(
                query_engine=full_summary_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        "about 'The Science of Computing an Introduction to Computer Science'"
                        "(a textbook for an inroductory computing course). For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            )
        ]

    llm = OpenAI(model="gpt-4")
    ch_agent = ReActAgent.from_tools(
            textbook_tools,
            llm=llm,
            verbose=True,
            system_prompt=f"""\
            You are a specialized agent designed to answer queries about the science of computing."
            You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
        """,
        )

    agents["Full"] = ch_agent
    query_engines["Full"] = full_index.as_query_engine(
            similarity_top_k=2
        )

    local_tool = QueryEngineTool(
            query_engine=agents["Full"],
            metadata=ToolMetadata(
                name=f"tool_fullText",
                description="This content contains 'The Science of Computing an Introduction to Computer Science'"
                "(a textbook for an inroductory computing course). Use"
                "this tool if you want to answer any questions about the science of computing.\n",
            ),
        )
    all_tools.append(local_tool)

    with open("exam_questions.txt") as f:
        prompts = f.readlines()

    nodes = []
    for idx, prompt in enumerate(prompts):
        node = TextNode(text = prompt, id = idx)
        nodes.append(node)
        
    exam_index = VectorStoreIndex(nodes, embed_model=embed_model)

    if not os.path.exists(f"./data/exam_questions.txt"):

        exam_index = VectorStoreIndex(nodes, embed_model=embed_model)
        exam_index.storage_context.persist(
            persist_dir=f"./data/exam_questions.txt"
            )
    else:
        exam_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/exam_questions.txt"),
            )

    exam_summary_index = SummaryIndex(nodes)

    exam_query_engine = exam_index.as_query_engine(llm=Settings.llm)
    exam_summary_engine = exam_summary_index.as_query_engine(llm=Settings.llm)

    exam_tools = [
        QueryEngineTool(
                query_engine=exam_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful as a reference for generating exam, test, or quiz questions. Contains an example of a test"
                    ),
                ),
            ),


        QueryEngineTool(
                query_engine=exam_summary_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic overview"
                        "of an exam on 'The Science of Computing an Introduction to Computer Science'"
                        "(a textbook for an inroductory computing course). For more specific examples of exam questions, please use the vector_tool."
                    ),
                ),
            )
        ]
    llm = OpenAI(model="gpt-4")
    exam_agent = ReActAgent.from_tools(
            exam_tools,
            llm=llm,
            verbose=True,
            system_prompt=f"""\
            You are a specialized agent designed to generate test questions about a given topic."
            You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
        """,
        )

    agents["exam"] = exam_agent
    query_engines["exam"] = exam_index.as_query_engine(
            similarity_top_k=2
        )


    exam_generation_tool = QueryEngineTool(
            query_engine=agents["Full"],
            metadata=ToolMetadata(
                name=f"tool_examGenerator",
                description="This content contains some useful examples of exams. Use"
                "this tool if you want to generate test questions about the science of computing.\n",
            ),
        )
    all_tools.append(exam_generation_tool)
   
    
    for idx, doc in enumerate(tqdm(documents)):
        
        nodes = LLAMA.split_document(doc)

        if not os.path.exists(f"./data/{doc}"):

            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(
                persist_dir=f"./data/{doc}"
            )
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"./data/{doc}"),
            )


        summary_index = SummaryIndex(nodes)

        vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
        summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name= "vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {doc.rsplit('.')[0]} (e.g. Data Structures, Algorithms, or Graphics)."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name= "summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f"  about {doc.rsplit('.')[0]}. For more specific questions about"
                        f" {doc.rsplit('.')[0]}, please use the vector_tool."
                    ),
                ),
            ),
        ]

        function_llm = OpenAI(model="gpt-4")
        # agent = ReActAgent.from_tools(    )
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
            system_prompt=f"""\
        You are a specialized agent designed to answer queries about {doc.rsplit('.')[0]}.
        You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
        """,
        )

        agents[doc] = agent
        query_engines[doc] = vector_index.as_query_engine(
            similarity_top_k=2
        )

    for doc in documents:
        _summary = (
            f"This content contains supplemental studying material on {doc.rsplit('.')[0]}. Use"
            f" this tool if you want to answer any questions about {doc.rsplit('.')[0]}.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=agents[doc],
            metadata=ToolMetadata(
                name=f"tool_{doc.rsplit('.')[0]}",
                description=_summary,
            ),
        )
        all_tools.append(doc_tool)

    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )

    top_agent = ReActAgent.from_tools(
        tool_retriever=obj_index.as_retriever(similarity_top_k=3),
        system_prompt=""" \
        You are an agent designed to answer queries about a variety of topics including different cities and a computing textbook.
        As well as perform calculations and answer general scientific questions using Wolfram Alpha API, get the current time or date.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        verbose=False,
    )

    return top_agent

def main(args):

    # if not os.path.exists(args.results_dir):
    #     os.makedirs(args.results_dir)

    # i = 0
    # filename = f"agent_exam_test_{i}.txt"
    # while(os.path.isfile(os.path.join(args.results_dir, filename))):
    #     i += 1
    #     filename = f"agent_exam_test_{i}.txt"

    
    # filepath = os.path.join(args.results_dir, filename)
    


    # with open(args.prompts) as f:
    #     prompts = f.readlines()
    
    agent = build_agents(args.doc_path)
    while True:
        user_prompt = input("Prompt: ") 
        if user_prompt.strip() == "": break    
        response = agent.query(user_prompt)

        print(response)

    # with open(filepath, 'w') as file:
    #     # for idx, prompt in enumerate(tqdm(prompts), start=1):
    #         for prompt in tqdm(prompts):
    #             file.write(f"Query: {prompt}\n\n")
    #             response = agent.query(prompt)

    #             file.write(f"Response: {response}\n\n\n")

            
    # print(f"Output written to {filepath}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--doc_path', type=str, default=None, nargs='+', help="Reference documents")
    parser.add_argument('--results_dir', type=str, default=os.path.join("responses"))
    parser.add_argument('--prompts', type=str, default=os.path.join("prompts", "exam_questions.txt"))


    main(parser.parse_args())