from chat_llm import ReflectingAgent
from txtai import Embeddings
from llm_engines import LLMApi, LLM
import json, random, logging,re
from jinja2 import Template, Environment, FileSystemLoader



# loading the prompt from a file
def load_dialogue_react_fewshots(file_path:str):
    ## dialogue react prompts are contained in jsonl
    with open(file_path, "r") as f:
        prompts= [json.loads(line) for line in f]
    return prompts

def load_dialogue_act_taxonomy(file_path:str):
    ## dialogue act taxonomy is contained in txt file
    with open(file_path, "r") as f:
        acts = f.readlines()
    return [act.strip() for act in acts]

def load_base_prompt(file_path:str)->Template:
    ## the base prompt is saved inside a jinja template called dialogue_react_generation.j2 in the prompts folder
    # load the file
    with open(file_path, "r") as f:
        template = f.read()
    return Template(template)

dialogue_acts=load_dialogue_act_taxonomy("prompts/dialogue_taxonomy.txt")
dialogue_react_fewshots = load_dialogue_react_fewshots("fewshots_mpc/generated_dialogue_reacts.jsonl")    
prompt_template = load_base_prompt("prompts/dialogue_react_generation.j2")



logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")


class DialogueReactAgent(ReflectingAgent):
    def __init__(self, name:str, llm:LLM, persona:str, memory_freq:int, reflections_freq:int, n_fewshots:int):
        """
        Initializes the DialogueReactAgent object.

        Args:
            name (str): The name of the agent.
            llm (LLM): The LLM object.
            persona (str): The persona of the agent.
            memory_freq (int): The frequency of memory updates.
            reflections_freq (int): The frequency of reflections.
            n_fewshots (int): The number of few-shot examples.

        Returns:
            None
        """
        
        self.llm = llm
        self.persona = persona
        self.memory_freq = memory_freq
        self.reflections_freq = reflections_freq
        self.n_fewshots = n_fewshots
        self.memory = Embeddings(content=True, gpu=False)
        self.prompt = prompt_template
        
        ## add the dialogue act taxonomy to the agent
        self.dialogue_acts = dialogue_acts
        
    def get_answer(self, last_messages, extra_context="", **kwargs):
        
        # unpack kwargs
        other_agents = kwargs["agent_list"]
        n_agents = kwargs["n_agents"]
        
        # if memory is empty, set optional_memory to ""
        if self.memory.isdense():
            print("Memory is not empty")
            pass ## todo: add memory component
        else:
            optional_memory = ""
        
        print(self.prompt.render(persona=self.persona, 
                                 last_messages=last_messages,
                                 extra_context=extra_context,
                                 other_agents=other_agents,
                                 n_agents=n_agents,
                                 dialogue_acts=self.dialogue_acts,
                                 optional_memory=optional_memory))
        

def main():
    # Create an instance of the DialogueReactAgent
    agent = DialogueReactAgent("Agent", LLMApi(), "John Doe", 0.5, 0.2, 5)

    # Perform some tests
    print(agent.llm)
    print(agent.persona)
    print(agent.memory_freq)
    print(agent.reflections_freq)
    print(agent.n_fewshots)
    print(agent.memory)

    agent.get_answer(["Hello", "How are you?"],
                     agent_list=["Agent1", "Agent2"],
                     n_agents=2)
    
    
if __name__ == "__main__":
    main()
        
        
        

        
    