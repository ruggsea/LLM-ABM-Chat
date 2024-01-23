
from llm_engines import LLMApi
import random, textwrap
from txtai import Embeddings


class Agent:
    def __init__(self, name, llm, prompt="", interests=[], behavior=[]): 
        self.name = name
        if prompt == "":
            interests_str = ", ".join(interests)
            behavior_str = ", ".join(behavior)
            self.prompt = f"""
            You are a person called {self.name}. You are texting with some friends in a telegram group. 
            You have interests in {interests_str}. You usually show the following attitude: {behavior_str}. 
            Keep your answers short and casual and do not sound too excited but try to keep the conversationg going as much as possible.
            Provide your answer as you would text it.
            """
            
            self.personal_prompt = f"""
            You are a person called {self.name}. You are texting with some friends in a telegram group. 
            You have interests in {interests_str}. You usually show the following behavior: {behavior_str}. 
            """
        else:
            self.prompt = prompt
            self.personal_prompt = prompt
        
        self.llm = llm
        self.llm.set_system_prompt(self.prompt)
        self.agent_answers = []
    
    
    def get_name(self):
        return self.name
    
    def get_answer(self, last_messages, extra_context=""):
        # ask the agent to answer to the chat, given the last messages
        
        formatted_last_messages_str = []
        last_messages = last_messages[-2:]
        for agent, message in last_messages:
            formatted_last_messages_str.append(f"{agent}: {message}")
            # make it one string
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        
        
        
        seek_answer_prompt = f"""The last messages were: {formatted_last_messages_str}. How do you answer? Your answer:
        {self.name}:"""
        
        # append system prompt to the prompt
        seek_answer_prompt = f"{self.prompt}\n{seek_answer_prompt}"
        

        if extra_context != "":
            # add extra context at the beginning
            seek_answer_prompt = f"{extra_context}\n{seek_answer_prompt}"
        
        agent_answer = ""
        
        # if answer is empty, try again
        while len(agent_answer)<5:
            agent_answer = self.llm.generate_response(seek_answer_prompt)
            
            
        # saving agent answer to agent personal answer list
        self.agent_answers.append(agent_answer)
        return agent_answer
    
    # create dummy routines for the agent
    def run_routines(self, turn_count, chat_history):
        # nothing to do here
        pass    
    
        
        



# make a thread class that make a chat thread with different agents

class ChatThread:
    def __init__(self, agent_list=[], chat_goal="", neutral_llm=LLMApi(), sel_method="random"):
        self.chat_history = []
        self.agent_list = agent_list
        if chat_goal == "":
            self.chat_goal = "There is not a specific goal for this conversation."
        else:
            self.chat_goal = chat_goal
            
        # define ways to start a conversation
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]
        self.neutral_llm = neutral_llm
        self.sel_method = sel_method
        
    def pick_random_agent(self):
        # pick a random agent
        random_agent = random.choice(self.agent_list)
        return random_agent
    
    def start_conversation(self):
        # start a conversation with a random agent
        
        # make sure the chat history is empty
        assert len(self.chat_history) == 0
        
        # choose a random agent        
        random_agent = self.pick_random_agent()
        # choose a random conversation starter
        random_conversation_starter = random.choice(self.conversation_starters)
        start_message= (random_agent.get_name(), random_conversation_starter)
        self.chat_history.append(start_message)
        return start_message
    
    def get_chat_answer(self, last_messages, agent):
        # get an answer from an agent
        
        other_agents = self.agent_list.copy()
        other_agents.remove(agent)
    
        other_agents_str = ", ".join([agent.get_name() for agent in other_agents])
        
        extra_context=f"""
        {self.chat_goal}. In this chat, there's you, {agent.get_name()}, and {other_agents_str}.
        """
        
        agent_answer = agent.get_answer(last_messages)
        self.chat_history.append((agent.get_name(), agent_answer))
        return agent_answer
    
    def render_last_message(self):
        # render the last message in the chat history
        last_message = self.chat_history[-1]
        agent, message = last_message[0], last_message[1]
        
        msg_string = f"\033[91m{agent}\033[0m: {message}"  # Set agent name to red
        
        # Wrap the message to a fixed width with reduced padding for new lines
        wrapped_message = textwrap.fill(msg_string, width=80, subsequent_indent=' ' * 4)
        
        # Display the message as raw text
        print(wrapped_message)
        
    def run_chat(self, max_turns=50):
        # run a chat with a random agent
        # start a conversation
        self.start_conversation()
        self.render_last_message()

        starting_agent = self.chat_history[-1][0]
        prev_agent = starting_agent
        
        # make a global variable to keep track of the turn count
        global turn_count
        turn_count = len(self.chat_history)
        
        while turn_count <= max_turns:
            # run agent routines (update memory, crearing reflections, etc.)
            for agent in self.agent_list:
                agent.run_routines(turn_count, self.chat_history)
            
            
            if self.sel_method == "random": ## to specify how to select the next agent
                # making sure the same agent is not selected twice in a row
                random_agent = self.pick_random_agent()
                while random_agent == prev_agent:
                    random_agent = self.pick_random_agent()

            # get last messages
            
            last_messages = self.chat_history

            # get an answer from the agent
            self.get_chat_answer(last_messages, random_agent)
            self.render_last_message()
            prev_agent= random_agent
            # increment turn count
            turn_count += 1
            
        # return the chat history
        return self.chat_history
    
    
# More advanced agents

# an agent with memory

class MemoryAgent(Agent):
    # an agent that abstracts observations of the chat and saves them in memory. 
    # It uses the memory to generate answers. 
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10):
        super().__init__(name, llm, prompt, interests, behavior)
        self.memory = Embeddings(content=True, gpu=False)
        self.n_last_messages = n_last_messages
        
    def get_observations(self, last_messages):
        # get observations from the last messages
        
        # format last messages
        formatted_last_messages_str = []
        
        for agent, message in last_messages[-self.n_last_messages:]:
            formatted_last_messages_str.append(f"{agent}: {message}")
            # make it one string
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        observation_prompt = """
        Your goal is to remember what happened in the last messages by saving memorable observations in your memory.
        Give me 5 personal observations about the last messages, divided by a new line.
        Those are the last messages in the chat:
        {formatted_last_messages_str}
        """
        
        # check if the answer contains 5 observations divided by a new line
        observations = []
        while len(observations) != 5:
            observations = self.llm.generate_response(observation_prompt)
            observations_list = observations.split("\n")

        
        return observations_list
    
    def save_observations(self, observations_list):
        # save observations in memory
        self.memory.upsert(observations_list)
        

    def get_answer(self, last_messages, extra_context=""):
        # get an answer from the agent enriching the context with observations from memory
        
        # get observations from memory relevant to the last message, if memory is not empty
        if self.memory.isdense():
            last_message= last_messages[-1]
            memories= self.memory.search(last_message[1], limit=5)
            context= "\n".join(x["text"] for x in memories)
        
        
        formatted_last_messages_str = []
    
        
        for agent, message in last_messages:
            formatted_last_messages_str.append(f"{agent}: {message}")
            # make it one string
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        
        seek_answer_prompt = f"""The last messages in the chat were: {formatted_last_messages_str}. How do you answer? Your answer:
        {self.name}:"""
        
        # append system prompt to the prompt
        seek_answer_prompt_with_context = f"""{self.prompt}\n
        In addition, you remember the following memories relevant to chat right now:
        {context}\n 
        {seek_answer_prompt}"""
        

        if extra_context != "":
            # add extra context at the beginning
            seek_answer_prompt = f"{extra_context}\n{seek_answer_prompt_with_context}"
        
        agent_answer = ""
        
        # if answer is empty, try again
        while len(agent_answer)<5:
            agent_answer = self.llm.generate_response(seek_answer_prompt)
            
            
        # saving agent answer to agent personal answer list
        self.agent_answers.append(agent_answer)
        return agent_answer

    


    def run_routines(self, turn_count, chat_history):
        # every n_last_messages turns, save observations in memory
        if turn_count % self.n_last_messages == 0:
            self.save_observations(self.get_observations(chat_history))
        else:
            pass

    
    
        
    
        