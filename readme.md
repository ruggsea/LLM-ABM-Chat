# LLM-ABM-Chat

This repository implements LLM (Language Learning Model) agents and chat objects to enable communication between agents. The agents are designed to be modular and the implemented LLM engine expects an OpenAI-like API at `http://127.0.0.1:1200`.

## Installation

To use this library, you need to have the following prerequisites installed:

- Python 3.x
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) with any instruct model loaded (or any OpenAI-like API) set up at `http://127.0.0.1:1200`
- in alternative, a gpt-3.5 model is provided as `ChatgptLLM` in `llm_engines.py`. It will try to load a OPENAI_API_KEY from a .env file in the root directory of the project. 

Once you have the prerequisites set up, you can install the library by following these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repo.git`
2. That's it, there are no requirements as of now.

## Usage

A jupyter notebook (`chat_interactive.ipynb`) is provided to demonstrate the usage of the library. 
The main functionality is provided by the `ChatThread` class, which can be used as follows:

```python
from chat_llm import ChatThread, Agent
from llm_engines import LLMApi

# defining two agents
John = Agent(name="Mario", llm=LLMApi(), interests=["hydraulics", "hats"], behavior="inspirational")
Mary = Agent(name="Luigi", llm=LLMApi(), interests=["ghosts", "pizza"], behavior="funny")

chat_test=ChatThread(agent_list=[John, Mary], neutral_llm=LLMApi())
```
After the chat thread is created, you can start the chat by calling the `run_chat` method:
```python
chat_test.run_chat()
```
The chat will run and print the messages to the console. At the end of the chat, the chat will be saved to a file in the `chat_logs` directory.

By default, the chat will run until the user stops it. You can also specify a maximum number of messages by passing the `max_messages` parameter to the `run_chat` method.
Furthermore, evaluation of the chat will be run by default, and the results will be saved to a file in the `chat_logs` directory. You can disable this by setting the `n_eval` parameter to -1.

## Types of agents

The library currently supports two types of agents:
- `Agent` - a simple agent that can be used to test the chat functionality
- `MemoryAgent` - an agent that generates observations and saves them to a vector store that serves as memory. Observations relevant to the last message in the chat are then used to generate responses.


## Planned features
- ~~memory system for agents~~ 
- dynamic summarization of the chat to fit model context length
- better logging
- better selection of the next agent to speak
- an automated way to end the chat based on the agents' behavior
