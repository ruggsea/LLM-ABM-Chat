# DialogueReact: Enhancing Conversation Synthesis for Social Science Simulations

This repository contains the codebase for my master thesis focused on developing DialogueReact, a novel framework for generating human-like dialogues suitable for complex social science simulations. The framework builds upon previous work in conversation synthesis by incorporating react prompting, dialogue acts, and agentic behaviour.

This codebase can also be used as a framework to experiment with agent-based modeling (ABM) simulations focused on conversations. It provides a flexible architecture for creating conversational agents with different capabilities (basic, memory-based, react-based) and managing multi-agent interactions through various LLM backends.

### Quick Example: Experimenting with Agent Behaviors

Here's a simple example of how to set up an experiment comparing different agent types:

```python
from dialogue_react_agent import DialogueReactAgent
from chat_llm import Agent, MemoryAgent
from llm_engines import LLMApi
from groupchat_thread import ChatThread
from chat_eval import calc_perplexity, calc_distinct_n

# Setup LLM backend
llm = LLMApi()  # or ChatgptLLM() for OpenAI

# Create different types of agents
basic_agent = Agent(name="Alice", llm=llm, 
                   interests=["art", "music"],
                   behavior="friendly")

memory_agent = MemoryAgent(name="Bob", llm=llm,
                          interests=["science", "books"],
                          behavior="analytical")

react_agent = DialogueReactAgent(name="Carol", llm=llm,
                                interests=["technology", "philosophy"],
                                behavior="inquisitive")

# Run conversations with different agent combinations
chat1 = ChatThread(agent_list=[basic_agent, memory_agent], neutral_llm=llm)
chat2 = ChatThread(agent_list=[memory_agent, react_agent], neutral_llm=llm)

# Generate conversations
conv1 = chat1.run_chat(max_messages=20)
conv2 = chat2.run_chat(max_messages=20)

# Analyze results
metrics1 = {
    'perplexity': calc_perplexity(conv1),
    'distinct1': calc_distinct_n(conv1, n=1),
    'distinct2': calc_distinct_n(conv1, n=2)
}

metrics2 = {
    'perplexity': calc_perplexity(conv2),
    'distinct1': calc_distinct_n(conv2, n=1),
    'distinct2': calc_distinct_n(conv2, n=2)
}

# Compare different agent combinations
print("Basic + Memory agents:", metrics1)
print("Memory + React agents:", metrics2)
```

## Prerequisites and Installation

### Prerequisites

To use this library, you need to have the following:

- Python 3.x
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) with any instruct model loaded (or any OpenAI-like API) set up at `http://127.0.0.1:1200`
- Alternatively, a GPT-3.5 model is provided as `ChatgptLLM` in `llm_engines.py`. It will try to load an `OPENAI_API_KEY` from a `.env` file in the root directory of the project.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in `.env` if using OpenAI's API

## LLM Engines

The framework supports multiple LLM backends through the `llm_engines.py` module:

- `LLMApi`: Connects to a local instance of oobabooga's text-generation-webui at `http://127.0.0.1:1200`
- `ChatgptLLM`: Uses OpenAI's GPT-3.5 API (requires API key in `.env`)

## Usage Examples

### Basic Chat Generation

```python
from places_replication import NaiveConversationAgent, NaiveConversationGeneration
from llm_engines import LLMApi

# Create agents with personas
agents = [
    NaiveConversationAgent("Alice", persona="She likes cats and philosophy of language."),
    NaiveConversationAgent("Bob", persona="He likes dogs and classical opera.")
]

# Initialize generator with chosen LLM
generator = NaiveConversationGeneration(agents, neutral_llm=LLMApi())

# Generate conversation with minimum turns
conversation = generator.generate_conversation(min_turns=15)
```

### Working with Chat History

```python
from chat_eval import load_chat_history, calc_perplexity, calc_distinct_n

# Load a previous chat
chat_history = load_chat_history("chat_history/naive_chat_history_chat_1719249634.json")

# Calculate metrics
perplexity = calc_perplexity(chat_history)
distinct_1 = calc_distinct_n(chat_history, n=1)  # Distinct-1 metric
distinct_2 = calc_distinct_n(chat_history, n=2)  # Distinct-2 metric
```

### Advanced Usage

For more advanced usage, including DialogueReact agents, memory-based agents, and group chats, refer to:
- `dialogue_react_agent.py` for DialogueReact implementation
- `groupchat_thread.py` for multi-agent conversations
- `advanced_agent.py` for enhanced agent capabilities

## Agent Classes

The framework implements several types of agents with different capabilities:

- **Basic Agent** (`Agent` in `chat_llm.py`): Simple agent with basic chat capabilities
- **Memory Agent** (`MemoryAgent`): Agent with memory capabilities using vector stores
- **DialogueReact Agent** (`DialogueReactAgent`): Main implementation incorporating react prompting and dialogue acts
- **Advanced Agent** (`AdvancedAgent`): Extended agent with additional capabilities

## Thesis Replication Steps

To replicate the thesis results, follow these steps:

1. **Dataset Preparation**
   - Download the Message-Persona-Context (MPC) dataset using `benchmark_fewshots.py`
   - Run `build_fits_dataset.py` to download and prepare the FITS dataset
   - Use `name_dist_experiment.py` to generate name distributions
   - Execute `synthetise_fits_personas.py` to create agent personas
   - Run `synthetise_dialogue_react.py` to prepare dialogue examples
   - Use `few_shots_benchmark.ipynb` to analyze the MPC dataset

2. **Generation Phase**
   The generation process creates 100 comparable conversations for each approach:

   a) **Initial Setup**
   - 100 topics are randomly selected from the FITS dataset
   - For each topic, 2 interested personas are selected from the generated dataset
   - Names are generated for each persona pair

   b) **Baseline Generation** (`thesis_chat_generation_naive.ipynb`)
   - Generates one-line conversation descriptions from personas
   - Creates conversations using few-shot examples
   - Ensures minimum 10 turns per conversation
   - Regenerates failed conversations

   c) **Agentic Approaches** (`thesis_chat_generation_non_naive.py`)
   - Instantiates agents with generated personas and names
   - Starts with random predefined greetings
   - Generates up to 50 turns per conversation
   - Memory component uses n=10 for generation
   - Reflection component uses n=25 for generation
   
   d) **Cross-LLM Validation** (`thesis_chat_generation_non_naive_gemma2.py`)
   - Repeats the process using Gemma 2 27B
   - Uses same persona pairs and topics
   - Validates consistency across different LLMs

3. **Evaluation**
   - Execute `running_eval.py` for the main evaluation pipeline
   - Alternative: use `running_eval_gemma2.py` for Gemma2 evaluations
   - Analyze results using `exploring_eval_results.ipynb`

## Project Structure

### Core Components

- `dialogue_react_agent.py`: Implementation of the DialogueReact framework
- `chat_llm.py`: Core LLM chat implementation and basic agent classes
- `groupchat_thread.py`: Management of multi-agent conversations
- `agent_factory.py`: Factory pattern for agent creation
- `llm_engines.py`: LLM engine implementations

### Evaluation Framework

- `chat_eval.py`: Evaluation metrics and assessment tools
- `running_eval.py`: Main evaluation execution
- `test_*.py` files: Unit tests for components

### Analysis Notebooks

- `reading_chats.ipynb`: Analysis of generated conversations
- `exploring_eval_results.ipynb`: Evaluation metrics visualization
- `chat_interactive.ipynb`: Interactive DialogueReact testing
- `few_shots_benchmark.ipynb`: Few-shot learning analysis

### Data and Resources

- `prompts/`: React prompting and dialogue act templates
- `fits_personas/`: Generated agent personas
- `chat_logs/`: Generated conversations
- `chat_history/`: Historical dialogue data
- `example_messages.json`: Training dialogue dataset
- `names_dist.jsonl`: Name distribution data

### Generation Scripts

- `thesis_chat_generation_*.py`: Main generation implementations
- `synthetise_*.py`: Persona and dialogue pattern generation

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in `.env`
4. Follow the replication steps above

## Key Features

- DialogueReact framework implementation
- React prompting and dialogue acts integration
- Multiple agent architectures
- Comprehensive evaluation metrics
- Interactive dialogue generation
- Extensive logging and analysis tools

## Logging

The system generates various logs:
- `chat.log`: Generated conversations
- `evaluation.log`: Evaluation metrics
- `agent_internal.log`: Agent reasoning states
- `running_eval.log`: Evaluation process
- `litellm.log`: LLM interactions

## Testing

The framework includes tests for:
- DialogueReact components
- Agent behavior and interactions
- Dialogue generation quality
- React prompting effectiveness

## Additional Features

### Conversation Stream Viewer

The project includes a retro-style conversation viewer (`conversation_stream.py`) that provides a visual way to review generated conversations:

```python
python conversation_stream.py
```

Features:
- Matrix-style terminal interface with green text on black background
- Load and stream multiple JSON conversation files
- Character-by-character streaming animation
- Alternating colors for different speakers
- Spacebar to skip to next conversation
- ESC to exit fullscreen mode

### Name Distribution Experiment

The project includes a name generation analysis tool (`name_dist_experiment.py`) that studies the relationship between personas and generated names:

```python
python name_dist_experiment.py
```

This experiment:
- Takes 50 random personas
- Generates 100 names for each persona
- Tracks name frequency distribution
- Saves results to `names_dist.jsonl`
- Helps understand LLM name generation patterns and biases

The results can be analyzed using `name_analysis.ipynb` to understand patterns in how LLMs associate names with different personas.
