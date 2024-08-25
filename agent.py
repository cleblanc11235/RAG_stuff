from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool, FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.chat_models import ChatOpenAI
from llama_index.core.callbacks import CallbackManager
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.objects import ObjectIndex
from llama_index.core.schema import IndexNode
from llama_index.core.llms import ChatMessage
from langchain.agents import initialize_agent
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    StructuredPlannerAgent,
    ReActAgentWorker,
    ReActAgent
)
from llama_index.core import (
    load_index_from_storage,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    VectorStoreIndex, 
    StorageContext,
    SummaryIndex, 
    Settings
)
from datetime import datetime
from pathlib import Path
import nest_asyncio
nest_asyncio.apply()
import requests
import base64
import dotenv
import LLAMA
import json
import os

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import google.auth



# If modifying these SCOPES, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]

def authenticate_gmail():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'creds.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_emails(input: str) -> str:
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    
    # Call the Gmail API to fetch the list of messages
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    if not messages:
        return 'No messages found.'
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            msg_snippet = msg['snippet']
            return (f'Message snippet: {msg_snippet}')


def send_email(input: str):
    # Parse the email string
    try:
        recipient, subject, body = input.split('|', 2)
        email_data = {
            'to': recipient.strip(),
            'subject': subject.strip(),
            'message_text': body.strip()
        }
    except ValueError:
        raise ValueError("Input string must be in the format: 'Recipient|Subject|Body'")

    # Authenticate and send the email
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    
    message = MIMEText(email_data['message_text'])
    message['to'] = email_data['to']
    message['subject'] = email_data['subject']
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    body = {'raw': raw_message}
    try:
        message = service.users().messages().send(userId='me', body=body).execute()
        print(f"Message Id: {message['id']}")
        return message
    except Exception as error:
        print(f"An error occurred: {error}")
        return None


def q_wolfram_alpha(input: str) -> str:
    """
    Queries the Wolfram Alpha LLM API with the given query string.

    Args:
        query (str): The query string to be sent to Wolfram Alpha.
        app_id (str): Your Wolfram Alpha API key.

    Returns:
        str: The response from Wolfram Alpha.
    """
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    params = {
        "input": input,
        "appid": "X9AAXJ-3PR8Q3L3GK"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

import sympy as sp
def evaluate_expression(input: str):
    expr = sp.sympify(input)
    steps = []

  
    steps.append(f"Original expression: {input}")
    simplified_expr = sp.simplify(expr)
    steps.append(f"Simplified expression: {simplified_expr}")

    result = expr.evalf()
    steps.append(f"Result: {result}")

    return "\n".join(steps)

def get_current_time(input=''):
    now = datetime.now()
    if 'date' in input.lower():
        return now.strftime("%Y-%m-%d")
    elif 'time' in input.lower():
        return now.strftime("%H:%M:%S")
    else:
        return f"Current Date and Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"





os.environ["OPENAI_API_KEY"] = "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"

# Wikipedia Articles to build agents
wiki_titles = [
    "Shreveport",
    "Joe_Biden",
    "Louisiana_Tech_University",
    "Cats",
    "Tokyo",
    "The_Talking_Heads",
    "Guitars",
    "Paris",
    "London",
    "Star_Trek",
    "Munich",
    "Golden_Retrievers",
    "Copenhagen",
    "llamas",
    "Cairo",
    "Palestine",
]


# extract page text
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# load data into list of Document objects 
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()


# Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
# Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", max_length=512)



node_parser = SentenceSplitter()

# create dictionaries of agents and query engines
# and list of tools
agents = {}
query_engines = {}

all_tools = []

# Creating Chapter 15 tools

chapter_nodes = LLAMA.split_document("Chapter_15.docx")
chapter_index = VectorStoreIndex(chapter_nodes)

if not os.path.exists(f"./data/Chapter_15"):

    chapter_index = VectorStoreIndex(chapter_nodes)
    chapter_index.storage_context.persist(
        persist_dir=f"./data/Chapter_15"
        )
else:
    chapter_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=f"./data/Chapter_15"),
        )

ch_summary_index = SummaryIndex(chapter_nodes)

chapter_query_engine = chapter_index.as_query_engine(llm=Settings.llm)
ch_summary_engine = ch_summary_index.as_query_engine(llm=Settings.llm)

chapter_tools = [
    QueryEngineTool(
            query_engine=chapter_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for questions related to specific aspects of"
                    "the science of computing."
                ),
            ),
        ),


    QueryEngineTool(
            query_engine=ch_summary_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Useful for any requests that require a holistic summary"
                    "of EVERYTHING about 'Chapter 15: The Past, Present, and Future of Computing'"
                    "(a textbook for an inroductory computing course). For questions about"
                    " more specific sections, please use the vector_tool."
                ),
            ),
        )
    ]

ch_llm = OpenAI(model="gpt-3.5-turbo")
ch_agent = OpenAIAgent.from_tools(
        chapter_tools,
        llm=ch_llm,
        verbose=True,
        system_prompt=f"""\
        You are a specialized agent designed to answer queries about the science of computing."
        You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """,
    )

agents["Chapter_15"] = ch_agent
query_engines["Chapter_15"] = chapter_index.as_query_engine(
        similarity_top_k=2
    )

local_tool = QueryEngineTool(
        query_engine=agents["Chapter_15"],
        metadata=ToolMetadata(
            name=f"tool_Chapter_15",
            description="This content contains 'Chapter 15: The Past, Present, and Future of Computing'"
            "(a textbook for an inroductory computing course). Use"
            "this tool if you want to answer any questions about the science of computing.\n",
        ),
    )
all_tools.append(local_tool)

for idx, wiki_title in enumerate(wiki_titles):
    nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])

    if not os.path.exists(f"./data/{wiki_title}"):

        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(
            persist_dir=f"./data/{wiki_title}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
        )


    summary_index = SummaryIndex(nodes)

    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" {wiki_title} (e.g. the history, arts and culture,"
                    " sports, demographics, or more)."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Useful for any requests that require a holistic summary"
                    f" of EVERYTHING about {wiki_title}. For questions about"
                    " more specific sections, please use the vector_tool."
                ),
            ),
        ),
    ]

    function_llm = OpenAI(model="gpt-3.5-turbo")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
    You are a specialized agent designed to answer queries about {wiki_title}.
    You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """,
    )

    agents[wiki_title] = agent
    query_engines[wiki_title] = vector_index.as_query_engine(
        similarity_top_k=2
    )



for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        f" this tool if you want to answer any questions about {wiki_title}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[wiki_title],
        metadata=ToolMetadata(
            name=f"tool_{wiki_title}",
            description=wiki_summary,
        ),
    )
    all_tools.append(doc_tool)





expression_tool = FunctionTool(
    fn=evaluate_expression,
    metadata=ToolMetadata(
        name="tool_ExpressionEvaluator",
        description="Useful for evaluating and explaining arithmetic expressions step-by-step",
    )
)

time_tool = FunctionTool(
    fn=get_current_time,
    metadata=ToolMetadata(
        name="tool_DateTime",
        description="Useful for getting the current time or date")
    )

wolfram_tool = FunctionTool(
    fn=q_wolfram_alpha,
    metadata=ToolMetadata(
        name="tool_WolframAlphaLLM",
        description="Useful for mathematical calculations and other general scientific questions"
    )
)
send_email_tool = FunctionTool(
    fn=send_email,
    metadata=ToolMetadata(
        name="tool_Send_Emails",
        description="Useful for sending emails in the format input = 'recipient@example.com|Subject|Body'")
    )

read_email_tool = FunctionTool(
    fn=get_emails,
    metadata=ToolMetadata(
        name="tool_Read_Emails",
        description="Useful for getting and reading email"
    )
)



all_tools.extend([time_tool, wolfram_tool, read_email_tool, send_email_tool])

llm = OpenAI(model="gpt-3.5-turbo-instruct", system_prompt=(
    "You are a helpful assistant designed to answer a user's query."
    "Use any of the tools available to you but DO NOT rely on prior knowledge!"
        )
    )




obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

# top_agent = ReActAgent.from_tools(
#     tool_retriever=obj_index.as_retriever(similarity_top_k=3),
#     system_prompt=""" \
#     You are an agent designed to answer queries about a variety of topics including different cities and a computing textbook.
#     As well as perform calculations and answer general scientific questions using Wolfram Alpha API, get the current time or date.
#     Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

# """,
#     verbose=True,
# )
# top_agent.query("What is the square root of 2?")

worker = FunctionCallingAgentWorker.from_tools(
    tools=all_tools,
    # tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    verbose=True
)

agent = StructuredPlannerAgent(
    worker,
    llm=llm,
    # tools=all_tools,
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    verbose=True
)


# # Summarize chapter 15 and send the result to Chris at cpl010@latech.edu.
# # What is square root of 2?
# # Summarize Chapter 15. Include 5 questions on the material covered



while True:
    
    user_prompt = input("Prompt: ") 
    if user_prompt.strip() == "": break
    
    response = agent.chat(user_prompt)
    # print(response)
# while(True):
#         # print("="*args.viewport_width)

#         user_prompt = input("Prompt: ")

#         if user_prompt.strip() == "": break
#         # print("-"*args.viewport_width)

        
#         response = top_agent.query(user_prompt)



























    # plan_id = agent.create_plan(
    #     user_prompt,
    # )

    # plan = agent.state.plan_dict[plan_id]

    # # for sub_task in plan.sub_tasks:
    # #     print(f"===== Sub Task {sub_task.name} =====")
    # #     print("Expected output: ", sub_task.expected_output)
    # #     print("Dependencies: ", sub_task.dependencies)

    # next_tasks = agent.state.get_next_sub_tasks(plan_id)

    # for sub_task in next_tasks:
    #     print(f"===== Sub Task {sub_task.name} =====")
    #     print("Dependencies: ", sub_task.dependencies)


    # for sub_task in next_tasks:
    #     response = agent.run_task(sub_task.name)
    #     agent.mark_task_complete(plan_id, sub_task.name)

    # while True:
    # # are we done?
    #     next_tasks = agent.get_next_tasks(plan_id)
    #     if len(next_tasks) == 0:
    #         break
    #     # print("TEEESSSST")
    #     # run concurrently for better performance
    #     responses = [agent.run_task(task_id) for task_id in next_tasks]
    #     for task_id in next_tasks:
    #         agent.mark_task_complete(plan_id, task_id)

    #     # refine the plan
    #     agent.refine_plan(
    #         user_prompt,
    #         plan_id,
    #     )
    #     # print(responses[0])






