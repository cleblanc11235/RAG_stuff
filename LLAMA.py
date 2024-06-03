import os
import openai
import dotenv

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import ( 
    RecursiveRetriever, 
    BaseRetriever,
    QueryFusionRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    load_indices_from_storage,
    Document,
    Settings,
    StorageContext,
    PromptTemplate,
    SimpleDirectoryReader,
    ChatPromptTemplate,
    QueryBundle,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
import json
from IPython.display import Markdown, display
from tqdm.asyncio import tqdm
from typing import List
import asyncio
from multiprocessing import Lock


if not dotenv.load_dotenv():
    print("Error: Must define a .env file")
    exit()

openai_key = os.getenv("OPENAI_API_KEY")
print("Openai Key:", openai_key)
DEBUG = True # '1' for debug, '0' for no debug
print("DEBUG:", DEBUG)


Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", max_length=512)
Settings.text_splitter = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

global embed_model
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", max_length=512)

def split_document(document_path: str) -> List[IndexNode]:
    """
    Split a document into individual nodes.

    Args:
        document_path: Path to the document.

    Returns:
        List of nodes.
    """

    if document_path.endswith(".pdf"):
        docs = PDFReader().load_data(file=document_path)
    else:
        docs = SimpleDirectoryReader(input_files=[document_path]).load_data()

    doc_text = "\n\n".join([d.get_content() for d in docs])

    nodes = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    ).get_nodes_from_documents([Document(text=doc_text)])

    for idx, node in enumerate(nodes):
        node.id_ = f"node-{idx}"

    return nodes


def ingest_document(document_path:str, vector_store):

    if document_path.endswith(".pdf"):
        loader = PDFReader()
        docs0 = loader.load_data(file=document_path)
        
    else:
        docs0 = SimpleDirectoryReader(input_files=[document_path]).load_data()

    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", max_length=512)

    pipeline = IngestionPipeline(
        transformations=[
            node_parser,
            embed_model,
        ],
        vector_store=vector_store,
        # cache=ingest_cache,
    )

    nodes = pipeline.run(documents=docs)
    for idx, node in enumerate(nodes):
        node.id_ = f"node--{idx}"



def initialize_index(document_path: str, force_reload: bool, collection_name: str) -> VectorStoreIndex:
    
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs

    # index_name = "./saved_indesx"
    client = chromadb.PersistentClient(path="./chroma_db")

    if force_reload:
        client.delete_collection(collection_name)
        documents = split_document(document_path)
        for idx, doc in enumerate(documents):
            doc.id_ = f"node--{idx}"

        collection = client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=vector_store,
            index_store=SimpleIndexStore()
        )
        index = VectorStoreIndex(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        
        # index.set_index_id("vector_index")
        # index.storage_context.persist()

    else:
        collection = client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        ingest_document(document_path, vector_store)
        index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model,
                )
        # with lock:
        #     if os.path.exists("storage"):
        #         index = load_index_from_storage(
        #             storage_context=StorageContext.from_defaults(
        #                 docstore=SimpleDocumentStore(),
        #                 vector_store=vector_store,
        #                 index_store=SimpleIndexStore(),
        #                 # persist_dir= "/home/chris/chatbot-rag-server/saved_index",
        #             ),
        #             # index_id="vector_index",
                    
        #         )
        #     else:
        #         index = VectorStoreIndex.from_vector_store(
        #             vector_store=vector_store,
        #             embed_model=embed_model,
        #         )
        #         index.storage_context.persist()

    return index


def create_docstore(document_path:str, docstore_path:str="./split_docstore"):
    
    nodes = split_document(document_path)

    docstore = SimpleDocumentStore(namespace="docstore")
    docstore.persist(persist_path=docstore_path)
    docstore.add_documents(nodes)


def get_retriever(document_path: str, collection_name:str = "supports", k:int = 5, force_reload:bool=False) -> BaseRetriever:

    index = initialize_index(document_path, force_reload, collection_name)

    return index.as_retriever(similarity_top_k=k)


def get_model(model_name: str, temperature=0, warmup=True):
    # If openai model, then langchain openai, else Chatollama
    openai_models = ["gpt-3.5-turbo", "gpt-4"]
    if model_name in openai_models:
        model = OpenAI(model_name=model_name, api_key=openai_key, temperature=temperature) 
        # Settings.llm = model
        return model
    else:
        model = Ollama(model=model_name, temperature=temperature, verbose=False)
        # Settings.llm = model
        if warmup: model.complete("") # Load model into memory
        return model

def get_expanded_retriever(retriever:BaseRetriever, model):

    # q_gen_prompt = get_expander_template()
    expanded_retriever = QueryFusionRetriever(
        [retriever],
        llm=model,
        similarity_top_k=2,
        num_queries=4,  # set this to 1 to disable query generation
        use_async=True,
        verbose=False,
        # query_gen_prompt=q_gen_prompt,
    )

    return expanded_retriever

def get_query_engine(base_retriever, model, prompt_template):

    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3
    )

    query_engine = RetrieverQueryEngine.from_args(
    retriever=base_retriever,
    llm=model,
    text_qa_template=prompt_template,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        rerank,
        MetadataReplacementPostProcessor(target_metadata_key="window"),
    ],
    streaming=True,
    # response_mode="tree_summarize",
    )

    return query_engine

def get_sme_template(system_prompt=""):

    qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
    )

    system_prompt = '''You are an expert on the past, present and future of computing.
        You communicate concisely and clearly.
        You will be given a user's query and some context.
        Briefly answer the user's query using exclusively the provided context.'''

    chat_text_qa_msgs = [
        ("system", system_prompt),
        ("user", qa_prompt_str),
    ]

    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    return text_qa_template


# results_dict = await run_queries(queries, [retriever1, retriever2])
async def run_expand(queries, retrievers):
    

    """Can take a list of retrievers as 2nd arg"""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict

# queries = generate_queries(llm, query_str, num_queries=4)
def generate_queries(llm, query_str: str, num_queries: int = 4):
    
    fmt_prompt = get_expander_template()

    fmt_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
        
    return queries

    
def get_expander_template() -> PromptTemplate:


    query_gen_prompt_str = (
        "You are a helpful assistant that generates multiple search queries based on a "
        "single input query. Generate {num_queries} search queries, one on each line, (DO NOT number the queries!!)"
        "related to the following input query:\n"
        "Query: {query}\n"
        "Queries:\n"
    )
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)

    return query_gen_prompt

# final_results = fuse_results(results_dict)
def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results.
    
    - Go through each node in each retrieved list, and add it's reciprocal rank to the node's ID. The node's ID is the hash of it's text for dedup purposes.
    - Sort results by highest-score to lowest.
    - Adjust node scores."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

# prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))

class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(
            self._llm, query_bundle.query_str, num_queries=4
        )
        results = asyncio.run(run_queries(queries, self._retrievers))
        final_results = fuse_results(
            results, similarity_top_k=self._similarity_top_k
        )

        return final_results
