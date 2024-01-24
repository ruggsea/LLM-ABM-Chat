
from llm_engines import LLMApi
import random, textwrap, logging
from txtai import Embeddings


logging.basicConfig(level=logging.INFO, filename="chat.log", filemode="w", format="%(asctime)-15s %(message)s")


class Agent:
    def __init__(self, name, llm, prompt="", interests=[], behavior=[]): 
        """
        Initializes an Agent object.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The LLM engine, it defaults to LLMAPI.
            prompt (str, optional): The initial prompt for the agent. If left empty, a modular prompt is generated using the interests and behavior arguments. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
        """
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
        """
        Returns the name of the agent.

        Returns:
            str: The name of the agent.
        """
        return self.name
    
    
    def log(self, prompt):
        """
        Logs a prompt for debugging answer generation.

        Args:
            prompt (str): The prompt to log.
        """
        logging.info(prompt)
        
    
    def get_answer(self, last_messages, extra_context=""):
        """
        Generates an answer from the agent given the last messages in the chat.

        Args:
            last_messages (list): The list of previous messages in the chat.
            extra_context (str, optional): Additional context for the agent's answer. Defaults to an empty string.

        Returns:
            str: The agent's answer.
        """
        formatted_last_messages_str = []
        last_messages = last_messages[-5:] # Only consider the last 5 messages  
        for agent, message in last_messages:
            formatted_last_messages_str.append(f"{agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        seek_answer_prompt = f"""
        The last messages were:
        ###
        {formatted_last_messages_str}
        ###
        
        How do you answer? Your answer:
        ###
        {self.name}:"""
        
        seek_answer_prompt = f"{self.prompt}\n{seek_answer_prompt}"
        
        if extra_context != "":
            seek_answer_prompt = f"{extra_context}\n{seek_answer_prompt}"
        
        agent_answer = ""
        
        self.log(seek_answer_prompt)
        
        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(seek_answer_prompt)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
        
        self.agent_answers.append(agent_answer)
        return agent_answer
    
    def run_routines(self, turn_count, chat_history):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        pass


class ChatThread:
    def __init__(self, agent_list=[], chat_goal="There is not a specific goal for this conversation.", neutral_llm=LLMApi(), sel_method="random"):
        """
        Initializes a ChatThread object.

        Args:
            agent_list (list, optional): The list of agents participating in the chat. Defaults to an empty list.
            chat_goal (str, optional): The goal of the chat. Defaults to "There is not a specific goal for this conversation.".
            neutral_llm (LLMApi, optional): The language model API for LLM generation needs outside the scope of any agent. Defaults to LLMApi().
            sel_method (str, optional): The method for selecting the next agent to answer. Defaults to "random".
        """
        self.chat_history = []
        self.agent_list = agent_list
        self.chat_goal = chat_goal
            
        self.conversation_starters = ["Hi!", "Hello!", "How are you?", "How is it going?", "What's up?", "How are you doing?"]
        self.neutral_llm = neutral_llm
        self.sel_method = sel_method
        
    def pick_random_agent(self):
        """
        Picks a random agent from the agent list.

        Returns:
            Agent: The randomly selected agent.
        """
        random_agent = random.choice(self.agent_list)
        return random_agent
    
    def start_conversation(self):
        """
        Starts a conversation by picking a random agent and a random conversation starter.
        
        Returns:
            tuple: The start message of the conversation.
        """
        assert len(self.chat_history) == 0
        
        random_agent = self.pick_random_agent()
        random_conversation_starter = random.choice(self.conversation_starters)
        start_message = (random_agent.get_name(), random_conversation_starter)
        self.chat_history.append(start_message)
        return start_message
    
    def get_chat_answer(self, last_messages, agent):
        """
        Gets an answer from an agent.

        Args:
            last_messages (list): The list of last messages in the chat.
            agent (Agent): The agent providing the answer.

        Returns:
            str: The agent's answer.
        """
        other_agents = self.agent_list.copy()
        other_agents.remove(agent)
    
        other_agents_str = ", ".join([agent.get_name() for agent in other_agents])
        
        extra_context = f"""
        {self.chat_goal}. In this chat, there's you, {agent.get_name()}, and {other_agents_str}.
        """
        
        agent_answer = agent.get_answer(last_messages)
        self.chat_history.append((agent.get_name(), agent_answer))
        return agent_answer
    
    def render_last_message(self):
        """
        Renders the last message in the chat history.
        """
        last_message = self.chat_history[-1]
        agent, message = last_message[0], last_message[1]
        
        msg_string = f"\033[91m{agent}\033[0m: {message}"
        wrapped_message = textwrap.fill(msg_string, width=80, subsequent_indent=' ' * 4)
        print(wrapped_message)
        
    def run_chat(self, max_turns=50):
        """
        Runs a chat simulation for a maximum number of turns. As of now, the simulations uses a random agent selection method.

        Args:
            max_turns (int, optional): The maximum number of turns in the chat. Defaults to 50.

        Returns:
            list: The chat history.
        """
        self.start_conversation()
        self.render_last_message()
        max_turns = max_turns + len(self.chat_history)

        starting_agent = self.chat_history[-1][0]
        prev_agent = starting_agent
        
        global turn_count
        turn_count = len(self.chat_history)
        
        while turn_count <= max_turns:
            for agent in self.agent_list:
                agent.run_routines(turn_count, self.chat_history)
            
            if self.sel_method == "random":
                random_agent = self.pick_random_agent()
                while random_agent == prev_agent:
                    random_agent = self.pick_random_agent()

            last_messages = self.chat_history

            self.get_chat_answer(last_messages, random_agent)
            self.render_last_message()
            prev_agent = random_agent
            turn_count += 1
            
        return self.chat_history
    

class MemoryAgent(Agent):
    def __init__(self, name, llm, prompt="", interests=[], behavior=[], n_last_messages=10):
        """
        Initializes a Agent object with memory.

        Args:
            name (str): The name of the agent.
            llm (LLMApi): The language model API.
            prompt (str, optional): The initial prompt for the agent. Defaults to an empty string.
            interests (list, optional): The interests of the agent. Defaults to an empty list.
            behavior (list, optional): The behavior of the agent. Defaults to an empty list.
            n_last_messages (int, optional): The number of messages that gets considered when generating memories. Defaults to 10.
        """
        super().__init__(name, llm, prompt, interests, behavior)
        self.memory = Embeddings(content=True, gpu=False)
        self.n_last_messages = n_last_messages
        
    def get_observations(self, last_messages):
        """
        Gets observations from the last messages.

        Args:
            last_messages (list): The list of last messages in the chat.

        Returns:
            list: The observations.
        """
        formatted_last_messages_str = []
        print(f"Processing observations for {self.name}...")

        for agent, message in last_messages[-self.n_last_messages:]:
            formatted_last_messages_str.append(f"{agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        observation_prompt = """
        Your goal is to remember what happened in the last messages by saving memorable observations in your memory.
        Give me 5 personal observations in third person about the other agents from the last messages, divided by a new line.
        Those are the last messages in the chat:
        ###
        {formatted_last_messages_str}
        ###
        """

        observations_list = []
        while len(observations_list) < 5:
            try:
                observations = self.llm.generate_response(observation_prompt)
                observations_list = observations.split("\n")
            except Exception as e:
                print(f"Observations could not be generated: {e}. Retrying...")

        return observations_list
    
    def save_observations(self, observations_list):
        """
        Saves observations in memory.

        Args:
            observations_list (list): The list of observations to save.
        """
        self.memory.upsert(observations_list)
        
    def get_answer(self, last_messages, extra_context=""):
        """
        Generates an answer from the agent enriching the context with observations from memory.

        Args:
            last_messages (list): The list of last messages in the chat.
            extra_context (str, optional): Additional context for the agent's answer. Defaults to an empty string.

        Returns:
            str: The agent's answer.
        """
        formatted_last_messages_str = []
        
        for agent, message in last_messages[-5:]:
            formatted_last_messages_str.append(f"{agent}: {message}")
        formatted_last_messages_str = "\n".join(formatted_last_messages_str)
        
        seek_answer_prompt = f"""
        The last messages in the chat were: 
        ###
        {formatted_last_messages_str}
        ###
        How do you answer? Your answer:
        ###
        {self.name}:
        """
        
        if self.memory.isdense():   
            last_message = last_messages[-1]
            memories = self.memory.search(last_message[1], limit=5)
            context = "\n".join(x["text"] for x in memories)
            seek_answer_prompt_with_context = f"""
            {self.prompt}\n
            In addition, you remember the following memories relevant to chat right now:
            {context}\n 
            {seek_answer_prompt}
            """
        else:
            seek_answer_prompt_with_context = f"""
            {self.prompt}\n
            {seek_answer_prompt}            
            """
        
        if extra_context != "":
            seek_answer_prompt = f"{extra_context}\n{seek_answer_prompt_with_context}"
        
        agent_answer = ""
        
        self.log(seek_answer_prompt_with_context)

        while len(agent_answer) < 5:
            agent_answer = self.llm.generate_response(seek_answer_prompt_with_context)
            if agent_answer == "":
                print("Empty answer returned. Retrying...")
        
        self.agent_answers.append(agent_answer)
        return agent_answer
    
    def run_routines(self, turn_count, chat_history):
        """
        Runs routines for the agent. It will be called by the ChatThread object at the end of every turn.

        Args:
            turn_count (int): The current turn count.
            chat_history (list): The chat history.
        """
        if turn_count % self.n_last_messages == 0:
            print(f"Time to generate memories for {self.name}!")
            self.save_observations(self.get_observations(chat_history))
        else:
            pass
    
    
