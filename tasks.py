from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

## researching Task
research_task = Task(
    description = (
        "Identify the video {topic}"
        "Get detailed information about the video from the channel"
    ),
    expected_output = "A comprehensive 5 paragraph (1st is introduction and last is conclusion) long report based on the {topic} of the video context", 
    tools = [yt_tool],
    agent = blog_researcher
)

## writing task
writing_task = Task(
    description = (
        "Get detailed information about the {topic} from the youtube channel"
    ),
    expected_output = "Summarize the information about the {topic} from the youtube channel and create the content for the topic", 
    tools = [yt_tool],
    agent = blog_writer,
    async_execution = False, ## If it is set to true the tasks will be executed parallely
    output_file = "new-blog-post.md"
)