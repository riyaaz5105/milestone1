# # LLM- Large Language Models

# # is a deep learning model trained on the vast amounts of the text data to:
# #     - uderstand the natural lang
# #     - generate the context-aware response
# #     - reasoning, summarize, classify, extract data, transfomr the content 
# #     - using the tools and API's (with the frameworks like LangChain)
    
    
# # Popular LLm's:
# #     - OpenAPI- GPT series(4o, 3.5)
# #     - Google- Gemini
# #     - Meta- Llama 3
# #     - Mistral AI- Mixtral 
# #     - Anthropic- Claude 3


# # LLM can be called without the LangChain 

# # useing OPenAPI (4o):
    
# # pip install openai python-dotenv 

# # .env -> storing your API_keys/ endpoints securely
# # OPENAI_API_KEY="your_openai_api_key"
# # OPENAI_ENDPOINT="your_openai_endpoint"

# # import os 
# # from openai import OpenAI 
# # from dotenv import load_dotenv

# # # 1. Load the API keys and endpoint from .env file
# # load_dotenv() # loading the .env file -> fetches the API key
# # client= OpenAPI(api_key= os.getenv("OPENAI_API_KEY"), api_base= os.getenv("OPENAI_ENDPOINT"))
# # # clinet->create a connection to the LLM service


# # # 2. Send a message to LLM (GPT-4o) and get the response
# # response= client.chat.completions.create(
# #     model= "gpt-4o-mini", # fast & cheap model for the demos 
# #     message=[
# #         {
# #             "role": "user",
# #             "content": "You are a helpful AI Assistant."
# #         }, 
# #         {
# #             " role": "user",
# #             "content": "Explain the machine learning in simple terms."
# #         }
# #     ]
# # )

# # # 3. print the result 

# # print(response.choices[0].message["content"]) ## LLM output


# # Basic LLM program:

# # pip install langchain langchain-community langchain-openai python-dotenv

# # import the packages
# from langchain_openai import ChatOpenAI

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# import os 
# from dotenv import load_dotenv  

# load_dotenv()

# # 1. connect to an LLM using Langchain Wrapper 

# llm = ChatOpenAI(
#     model= "gpt-4o-mini",
#     temperature= 0.2,
#     api_key= os.getenv("OPENAI_API_KEY")
# )

# # 2. create a prompt template 

# prompt= PromptTemplate(
#     input_variables= ["topic"],
#     template= """
#     You are a helpful AI assistant. Explain the topic below in simple, easy-to-understand language:
    
#     Topic: {topic}
#     """
    
# )

# # 3. create a chain (LLM + prompt)

# chain = LLMChain(
#     llm= llm, 
#     prompt= prompt
# )

# #4. run the chain 

# result= chain.run({"topic" : "How machine learning works" })
# print(result)  # LLM output


##Zero- Shot Agent code 

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Dummy tool example
def greet(name: str):
    return f"Hello {name}, welcome to Agents!"

greet_tool = Tool(
    name="greeting_tool",
    func=greet,
    description="Use this to greet a person by name."
)

# Create a Zero-Shot ReAct Agent
agent = initialize_agent(
    tools=[greet_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("Greet Saadhana in a friendly way.")
print(result)

## Tool example: weather API tool

import requests

def get_weather(city: str):
    url = "https://wttr.in/{}?format=j1".format(city)
    data = requests.get(url).json()
    temp = data["current_condition"][0]["temp_C"]
    return f"Current temperature in {city} is {temp}°C"

weather_tool = Tool(
    name="weather",
    func=get_weather,
    description="Get current temperature of a city"
)
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


## memory- 3 tupes

# 1. short term memory

from langchain.memory import ConversationBufferMemory

memory= ConversationBufferMemory(
    memory_key= "chat_history")

# - used for: chatbots, agents with multi steps 

# 2. Window memory 

from langchain.memory import ConversationBufferWindowMemory

window_memory= ConversationBufferWindowMemory(
    memory_key= "chat_history")

# 3. Long term memory 

from langchain.memory import VectorStoreRetrieverMemory 

from langchain.vectorstores import FAISS

## full code- zero shot + tools + memory 

#packages: langchain, langchain-openai, langchain-community, python-dotenv, requests

# in your .env file- you will have the openai key stored securely

#OPENAI_API_KEY= "your_openai_api_key"


# build tools.py 

#tool 1- simple greeting tool 

def greet(name:str):
    return f"Hello {name}, I am your Langchain Agent!"

# tool2- weather API tool

import requests 


def weather(city:str):
    url = f"url = https://wttr.in/{city}?format=j1"
    res= requests.get(url).json()
    temp= res["current_condition"][0]["temp_C"]
    return f"Current temperature in {city} is {temp}°C"

# convert into langchain tools 

from langchin.tools import Tool

greet_tool= Tool(
    name= "greeting_tool",
    func= greet,
    description= "Use this to greet a person by name."
)

weather_tool = Tool(
    name= "weather",
    func= weather,
    description= "Get current temperature of a city"
)


# build memory.py 

from langchain.memory import ConversationBufferMemory

memory= ConversationBufferMemory(
    memory_key= "chat_history" 
    return_messages= True)

# build agents.py 

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# from tools import tools 
# from memory import memory 

import os 
from dotenv import load_dotenv

load_dotenv()

def create_agent():
    llm = ChatOpenAI(
        model= "gpt-4o-mini",
        temperature= 0,
        api_key= os.getenv("OPENAI_API_KEY")
    )
    
    agent= initialize_agent(
        tools= tools,
        llm= llm,
        agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory= memory,
        verbose= True
    )
    
    return agent

# build main.py 

from agents import create_agent

agent= create_agent()

print("Langchain Agent is ready!")
print("Type 'exit' to quit.")

while True:
    query= input ("you :")
    
    if query.lower() == "exit":
        break 
    
    response= agent.run(query)
    print("Agent:", response)