# %% [markdown]
# Trying to see what is the most optimal number of few shot prompts to use in the agent answer generation.

# %%
# create some partials for the agents 

from chat_llm import ReflectingAgent, ChatThread, load_chat
from llm_engines import LLMApi
from functools import partial



# %%
john_partial = partial(ReflectingAgent, name="John",llm=LLMApi(), interests=["gossip","f1", "climate change"], behavior=["smart","reflective","funny"], n_last_messages=10, n_last_messages_reflection=20)
mary_partial = partial(ReflectingAgent, name="Mary",llm=LLMApi(), interests=["bouldering", "italian movies", "playing the arp"], behavior=["kind", "creative", "friendly"], n_last_messages=10, n_last_messages_reflection=20)

# %%
# a list of number of examples for each agent
import os 

n_examples = [1,2,3,5,10,15,20,25]

for n in n_examples:
    john = john_partial(n_examples=n)
    mary = mary_partial(n_examples=n)
    # run 10 chat per configuration
    for i in range(10):
        chat = ChatThread(agent_list=[john,mary], neutral_llm=LLMApi(), n_eval=25)
        chat.run_chat(max_turns=75)
        
        try:
            # find latest chat in the chat_logs folder
            chat_logs = os.listdir("chat_logs")
            # add folder to the path
            chat_logs = [os.path.join("chat_logs",chat) for chat in chat_logs]
            chat_logs.sort(key=os.path.getmtime, reverse=True)
            latest_chat = chat_logs[0]
            
            print(f"Chat {i} with {n} examples completed. Chat log saved at {latest_chat}")
            
        except:
            continue



