from chat_llm import ReflectingAgent
from txtai import Embeddings
from llm_engines import LLMApi, LLM
import json, random, logging,re

# loading the prompt from a file
def load_prompt(file_path:str):
    with open(file_path, "r") as file:
        prompt = file.read()
    return prompt

with open("react_examples.json", "r") as f:
    react_examples = json.load(f)

logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")


class AdvancedAgent(ReflectingAgent):
    def __init__(self, name:str, llm:LLM=LLMApi(), n_last_messages:int=10, profile="", personal_goal="",n_last_messages_reflection:int=20, n_examples:int=3):
        self.name = name
        self.llm = llm
        self.n_last_messages = n_last_messages
        self.n_last_messages_reflection = n_last_messages_reflection
        self.n_examples = n_examples
        self.memory = Embeddings(content=True, gpu=False)
        self.profile = profile
        self.personal_goal = personal_goal
                
        
        # if personal goal is not provided, use the default
        if personal_goal == "":
            self.personal_goal = "I want to have a good conversation."
        else:
            self.personal_goal = personal_goal
    
        self.prompt_template = load_prompt("prompts/advanced_generation.txt")
        self.preprompt = load_prompt("prompts/advanced_pre-prompt_only.txt").format(name=self.name, profile=self.profile, personal_goal=self.personal_goal)

        self.agent_answers = []
        
        
        # in order to generate memories, a preprompt must be set
        self.prompt=self.preprompt
        
        
    def get_answer(self, last_messages, extra_context="", **kwargs):
        # unpack kwargs
        other_agents = kwargs["agent_list"]
        chat_goal = kwargs["chat_goal"]
        n_agents = kwargs["n_agents"]
                
        agent_list= f"You, {other_agents}"
        
        
        
        few_shot_examples = random.sample(react_examples, self.n_examples)
        
        # add quotes around the examples
        few_shot_examples = [f"'{x}'" for x in few_shot_examples]
        
        # join the examples into a string with two new lines between each example
        few_shot_examples_str = "\n\n".join(few_shot_examples)
    
            
        observation_thought_action_examples_list= few_shot_examples_str
        
        
        
        # chat history
        formatted_last_messages_str = []
        for message_n, agent, message in last_messages:
            formatted_last_messages_str.append(f"{message_n}. {agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        # check if memory is empty
        if self.memory.isdense():   
            last_message = last_messages[-1]
            memories = self.memory.search(last_message[-1], limit=5)
            memory_list_str = "\n".join(x["text"] for x in memories)
            memory_prompt= load_prompt("prompts/memory_prompt.txt")
            optional_memory = memory_prompt.format(memory_list_str=memory_list_str)
        else:
            optional_memory = ""
        
        # create the prompt
        prompt = self.prompt_template.format(
            name=self.name,
            profile=self.profile,
            n_agents=n_agents,
            agent_list=agent_list,
            chat_goal=chat_goal,
            personal_goal=self.personal_goal,
            chat_history=formatted_last_messages_str,
            observation_thought_action_examples_list=observation_thought_action_examples_list,
            optional_memory=optional_memory
            )
        

        # get the answer from the llm
        
        agent_answer = ""
        
        self.log(prompt)
        
        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(prompt)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
                
            # find the string between Action: and ## using regex
            pattern = r"Action: (.*?)##"
            matches = re.findall(pattern, agent_answer)
            if matches:
                agent_answer = matches[-1]
                agent_answer = agent_answer + "##"
            else:
                print("Invalid answer: no React style answer found. Retrying...")
                logging.info(f"Invalid answer: {agent_answer}.")
                agent_answer = ""
                continue
        
        self.agent_answers.append(agent_answer)
        
        # save the agent's answer to a store of React answers
        with open("react_answers.txt", "a") as f:
            f.write(agent_answer + "\n\n")
        
        return agent_answer
        
        
        
    def dump_agent(self):
        """
        Turn the agent into a dictionary for dumping into a json file.
        
        Returns:
            dict: The agent's data.
        """
        
        memory_dump = self.memory.search("SELECT id, text, turn, entry FROM txtai ORDER BY entry DESC", limit=100)

        
        agent_data = {"name": self.name, "prompt": self.prompt, "agent_answers": self.agent_answers , "type":self.__class__.__name__, "n_examples": self.n_examples,"llm": self.llm.__class__.__name__ , 
                      "personal_goal": self.personal_goal, "profile": self.profile, "n_last_messages": self.n_last_messages, "n_last_messages_reflection": self.n_last_messages_reflection, "memories": memory_dump}
        
        return agent_data
    
    