# 🎥 YouTube Video to Blog Writer Agent using CrewAI & Groq

An autonomous multi-agent system designed to search a target YouTube channel for a specific topic, extract semantic information from relevant videos, and write a structured, educational blog post. 

The project leverages the **CrewAI** framework for agent orchestration, sequential task execution, and model reasoning, powered by **Groq's LPU inference** for fast content generation.

---

## 🏗️ Multi-Agent Architecture

The pipeline divides the workflow between two specialized autonomous agents executing tasks in a sequential process:

```mermaid
graph TD
    A[Input Topic: AI VS ML VS DL VS Data Science] -->|Sequential Process| B[blog_researcher: Senior Content Researcher]
    B -->|Tools: YoutubeChannelSearchTool| C[@krishnaik06 Channel Videos]
    C -->|Extract Context & Details| B
    B -->|Output: 5-Paragraph Research Report| D[blog_writer: Senior Content Writer]
    D -->|Synthesize & Write| E[Output File: new-blog-post.md]
```

---

## 👥 Agents & Roles

### 1. Senior Blog Content Researcher (`blog_researcher`)
* **Role**: Blog researcher from YouTube Videos.
* **Goal**: Get the relevant video content for the specified topic from the configured YouTube channel.
* **Backstory**: Expert in analyzing videos in AI, Data Science, Computer Vision, NLP, Machine Learning, and Deep Learning.
* **Key Option**: `allow_delegation = True` (can delegate sub-questions or query tasks back to peer agents if needed).

### 2. Senior Blog Content Writer (`blog_writer`)
* **Role**: Blog Content Writer.
* **Goal**: Narrate compelling, educational tech stories about the researched topic.
* **Backstory**: Possesses a flair for simplifying complex concepts, crafting engaging narratives that educate and bring new discoveries to light.
* **Key Option**: `allow_delegation = False` (the final agent in the sequential chain, meaning it focuses solely on producing the final output).

---

## 📋 Tasks Defined

### 1. Research Task (`research_task`)
* **Description**: Identify the target video for the topic and extract detailed information from the channel.
* **Expected Output**: A comprehensive 5-paragraph long report (Introduction, 3-paragraph body, Conclusion) based on the video context.
* **Assigned Agent**: `blog_researcher`

### 2. Writing Task (`writing_task`)
* **Description**: Synthesize the detailed information gathered by the researcher and create the final structured blog content.
* **Expected Output**: A formatted markdown post summarizing the topic.
* **Assigned Agent**: `blog_writer`
* **Output File**: Automatically writes the final article to `new-blog-post.md`.

---

## 🛠️ Technology Stack

* **CrewAI**: Multi-agent framework that defines agents, assigns goals, and runs them sequentially (`Process.sequential`).
* **CrewAI Tools (`YoutubeChannelSearchTool`)**: Performs semantic similarity search within a targeted YouTube channel (set to `@krishnaik06` by default in `tools.py`).
* **Groq Cloud (via LangChain)**: Powered by LPU hardware to run the `gemma2-9b-it` model, enabling rapid reasoning and content synthesis.

---

## ⚙️ Setup and Installation

### Prerequisites
* Python 3.8+
* A Groq API Key (obtained from [Groq Console](https://console.groq.com/))
* An OpenAI API Key (optional but recommended, as `YoutubeChannelSearchTool` uses OpenAI Embeddings by default to index video transcripts for similarity search).

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root of the project directory and fill in your keys:

```env
# Groq API token for LLM reasoning
GROQ_API_KEY=gsk_your_groq_api_key

# OpenAI API key (required by the YoutubeSearchTool indexer)
OPENAI_API_KEY=sk-proj-your_openai_api_key
```

### 3. Run the Crew
Execute the script to start the sequential execution loop:
```bash
python crew.py
```
Upon completion, the researcher's report will be passed to the writer, and the final blog article will be generated as **`new-blog-post.md`**.
