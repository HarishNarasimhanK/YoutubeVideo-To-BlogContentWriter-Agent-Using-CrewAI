## Agent => people who does tasks
from crewai import Agent
from tools import yt_tool

## Creating a senior blog content researcher 
blog_researcher = Agent(
    role = "Blog researcher from youtube Videos",
    goal = "Get the relevant video content for the topic {topic} from Youtube Channel",
    name = "Senior Blog content researcher",
    description = "A senior blog content researcher for Youtube videos",
    verbose = True,
    memory = True,
    backstory = (
        "Expert in understanding videos in AI, DataScience, Computer vision, NLP, Machine learning and deep learning"
    ),
    tools = [yt_tool],
    allow_delegation = True
)


## Creating a senior blog content writer 
blog_writer = Agent(
    role = "Blog Content Writer",
    goal = "Narrate the compelling tech stories about the topic {topic}",
    name = "Senior Blog content Writer",
    description = "A senior blog Writer for Youtube videos",
    verbose = True,
    memory = True,
    backstory = (
        "With a fair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner"
    ),
    tools = [yt_tool],
    allow_delegation = False # last in agent chain so no delegations
)
