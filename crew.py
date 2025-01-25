from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import research_task, writing_task
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = groq_api_key
)

crew = Crew(
    llm = llm,
    agents = [blog_researcher, blog_writer],
    tasks = [research_task, writing_task],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm = 150, 
    share_crew = True 
)

## starting the task execution
result = crew.kickoff(inputs = {"topic" : "AI VS ML VS DL VS Data Science"})
print(result)