from llm_engines import LLMApi, ChatgptLLM, LLM
from chat_llm import Agent
from dialogue_react_agent import load_base_prompt
import random, logging, time,json, os, textwrap


logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")


# few-shot examples for the PLACES paper are contained in a jsonl file at prompts/places_examples.jsonl, let's load them, eacj line is a few-shot example conversation
def load_places_examples(file_path:str="prompts/places_examples.jsonl"):
    """
    Load the few-shot examples for the PLACES paper from a jsonl file.
    """
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            examples.append(json.loads(line)["conversation"])
    return examples



# load the few-shot examples
places_examples = load_places_examples()

# load the base prompt for the PLACES paper
naive_generation_prompt = load_base_prompt("prompts/naive_generation.j2")


class NaiveConversationAgent(Agent):
    """
    Naive conversation agent used for replication of the PLACES paper. It only has a name and a persona.
    """
    def __init__(self, name:str, llm:LLM=LLMApi(), persona:str=""):
        self.name = name
        self.persona=persona
        self.llm = llm
    
    def dump_agent(self):
        return {"name": self.name, "persona": self.persona}


class NaiveConversationGeneration:
    def __init__(self, agent_list:list=[], neutral_llm:LLM=LLMApi(),topics_to_cover:list=[]) -> None:
        """
        Initializes a NaiveConversationGeneration object, a naive generation method for conversations based on the PLACES paper. Takes a list of agents and topics to cover and uses them to generate a dialogue using a simple LLM call
        """
        self.agent_list = agent_list
        self.neutral_llm = neutral_llm
        self.topics_to_cover = topics_to_cover
        self.turn = 0
        self.generation_template=naive_generation_prompt
        self.few_shot_examples = places_examples
        
        
        # conversation starters
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]
        
        # conversation history
        
        self.chat_history = []
        
        # unique identifier for the chat
        self.chat_id = f"chat_{str(int(time.time()))}"
        
        # agent colors: each agent is assigned a different ansi color - a dict of agent_name: ansi_color
        
        self.agent_colors = {}
        for i, agent in enumerate(self.agent_list):
            agent_color = f"\033[9{i+2}m"
            agent.color = agent_color
            self.agent_colors[agent.name] = agent_color

    def print_chat_history(self):
        """
        Prints the chat history to the console.
        """
        for turn, agent_name, message in self.chat_history:
            agent_color = self.agent_colors[agent_name]
            msg_string =  f"{agent_color}{agent_name}\033[0m: {message}"
            wrapped_message = textwrap.fill(msg_string, width=80, subsequent_indent=" "*4)
            print(wrapped_message)
            
    def save_chat_history(self, folder:str="chat_history"):
        """Save the chat history to a JSON file."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/naive_chat_history_{self.chat_id}.json", "w") as f:
            json.dump(self.chat_history, f)
            
    def dump_chat(self):
        """
        Dumps the chat history to the console.
        """
        chat_data = {"chat_id": self.chat_id, "chat_history": self.chat_history, "agent_list": [agent.dump_agent() for agent in self.agent_list], "neutral_llm": self.neutral_llm.__class__.__name__, "topics_to_cover": self.topics_to_cover}
        if not os.path.exists("chat_logs"):
            os.makedirs("chat_logs")
        with open(f"chat_logs/{self.chat_id}.json", "w") as f:
            json.dump(chat_data, f)    
        
        
    def generate_conversation(self, min_turns :int=10, start_conversation:bool=True):
        """
        Generates a conversation between the agents in the agent list.
        
        Parameters:
        min_turns (int, optional): The minimum number of turns for the conversation. Defaults to 10.
       
        Returns:
            list: The chat history.
        """
        # clear the chat history
        
        self.chat_history = []
        
        if start_conversation:
            # select a random conversation starter
            conversation_starter = random.choice(self.conversation_starters)
            # select speaker for the conversation starter
            speaker = random.choice(self.agent_list)
            conversation_start_tuple=(0, speaker.name, conversation_starter)
            # add the conversation starter to the chat history
            self.chat_history.append(conversation_start_tuple)
            conversation_starter_string = f"{speaker.name}: {conversation_starter}"
        else:
            conversation_starter_string = ""
        

        
        
        # fill out the prompt template
        prompt = self.generation_template.render(
            agent_names=[agent.name for agent in self.agent_list],
            agent_personas=[agent.persona for agent in self.agent_list],
            conversation_starter_string=conversation_starter_string,
            first_speaker=speaker.name,
            few_shot_examples=random.sample(self.few_shot_examples,3)
        )
        
        logging.info(f"Naive generation prompt: {prompt}")
        
        # the answer should a list of strings, one message per line until the max_turns is reached
        
        conversation_len=0
        
        # keep generating until you have a conversation of min max_turns length
        
        while conversation_len < min_turns:
            conversation_len = 1 if start_conversation else 0
            
            # reset the chat history for each turn
            temp_chat_history = []
            
            # generate the answer
            answer = self.neutral_llm.generate_response(prompt)
            
            # split the answer into lines, each line should start with the agent name and a colon
            answer_lines = answer.split("\n")
            
            # add the answer to the chat history
            for line in answer_lines:
                # split the line into agent name and message only at the first colon
                try:
                    agent_name, message = line.split(":", 1)
                    if agent_name in [agent.name for agent in self.agent_list]:
                        temp_chat_history.append((self.turn, agent_name, message))
                        self.turn += 1
                        conversation_len += 1
                    else:
                        assert False, f"Invalid agent name in the answer: {agent_name}"                        
                except:
                    logging.info(f"Invalid line in the answer: {line}")
                    # finish the for loop
                    break
                
            if conversation_len < min_turns:
                logging.info(f"Conversation {self.chat_id} with {len(temp_chat_history)} turns generated, but it is too short.")                    
                
        logging.info(f"Conversation {self.chat_id} with {len(temp_chat_history)} turns generated.")
                
        # put the chat history in a list of (turn, agent_name, message) tuples
        temp_chat_history = [(turn, agent_name, message) for turn, agent_name, message in temp_chat_history]
        
        self.chat_history.extend(temp_chat_history)
        
        # save the chat history
        self.save_chat_history()
        
        # dump chat logs
        self.dump_chat()
        
        return self.chat_history
    
    
    
## debugging

if __name__=="__main__":
    # create a list of agents
    agent_list = [NaiveConversationAgent(name="Alice", persona="Alice likes cars"), NaiveConversationAgent(name="Bob", persona="Bob likes books")]
    # create a neutral LLM
    neutral_llm = ChatgptLLM()
    # create a NaiveConversationGeneration object
    naive_generation = NaiveConversationGeneration(agent_list=agent_list, neutral_llm=neutral_llm)
    # generate a conversation
    chat_history = naive_generation.generate_conversation(min_turns=10)
    # print the chat history
    naive_generation.print_chat_history()
    # save the chat history
    naive_generation.save_chat_history()
    # dump the chat history
    naive_generation.dump_chat()