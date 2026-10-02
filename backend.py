import os 

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict,Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph ,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

# database url...................
def get_database_url():
    database_url=os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE URL Is missing. please add the postgressql external database url"
        )
    if "sslmode=" not in database_url:
        separator= "&" if  "?" in database_url else "?"
        database_url=f"{database_url}{separator}sslmode=require"
    return database_url

GROQ_API_KEY= os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. please add it to your .env file")

#llm

llm=ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
    max_tokens=900,
)

class TravelState(TypedDict):
    messages :Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary: str
    llm_calls: int

def flight_agent(state:TravelState):
    query=state["user_query"]
    flight_data=search_flights(query)

    return {
        "flight_results":flight_data,
        "messages":[
            AIMessage(content="flight results fetched. ")

        ],
        "llm_calls":state.get("llm_calls",0)+1
    }

    #hotel agent.....


def hotel_agent(state:TravelState):
    query=f"best hotels for {state['user_query']}"
    hotel_results=tavily_search(query)

    return {
            "hotel_results":hotel_results,
            "messages":[
                AIMessage(content="hotel results fetched")
            ],
            "llm_calls":state.get("llm_calls",0)+1
        }


    #itenery agent......

def itinerary_agent(state:TravelState):
    prompt =f"""
    ceate a complete travel iteinerary.

    user query:
    {state['user_query']}

    flight results:
    {state['flight_results']} 

    hotel results:
    {state['hotel_results']}

    Make the itinerary practical, budget-aware, and easy to follow.
    Keep the complete answer concise and under 650 tokens. Include a trip
    summary, transport, lodging, a day-by-day outline, and estimated budget.
    Do not invent live flight schedules or ticket prices; state when those
    details are unavailable in the provided search results."""

    response =llm.invoke([
        SystemMessage(content="you are an expert travel planner"),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }



    # =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    return {
        "messages": [AIMessage(content=state["itinerary"])],
    }


# build graphs

graph=StateGraph(TravelState)

graph.add_node("flight_agent" ,flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_agent)


graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

#Postgresql checkPointer..................

DATABASE_URL=get_database_url()

_conn=psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer=PostgresSaver(_conn)
checkpointer.setup()

travel_graph=graph.compile(checkpointer=checkpointer)




# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }