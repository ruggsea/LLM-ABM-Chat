from dialogue_react_agent import DialogueReactAgent
from llm_engines import LLMApi
import json 
import random
import itertools

# set up logging
import logging

# Create a custom logger
angent_factory_logger = logging.getLogger('agent_factory')
angent_factory_logger.setLevel(logging.INFO)

# Create handlers
agent_factory_file_handler = logging.FileHandler('agent_factory.log', mode='w')
agent_factory_file_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)-15s %(message)s')
agent_factory_file_handler.setFormatter(formatter)

# Add handlers to the logger
angent_factory_logger.addHandler(agent_factory_file_handler)

def read_personas():
    list_personas=[]
    with open("fits_personas/generated_personas.jsonl", "r") as f:
        for line in f:
            persona = json.loads(line)
            list_personas.append(persona)
    return list_personas


def read_topics():
    with open("fits_personas/topics.json", "r") as f:
        topics=json.load(f)
        
    return topics
    
personas=read_personas()

unique_topics_domains=read_topics()

unique_topics=[topic["generic_topic"] for topic in unique_topics_domains]
unique_domains=[topic["domain"] for topic in unique_topics_domains]


def validate_topics(topics:list):
    for topic in topics:
        if topic not in unique_topics:
            raise ValueError(f"Invalid topic: {topic}")


def gen_name(persona, neutral_llm=LLMApi()):
    name=""
    while len(name)<2 or len(name)>20:
        name=neutral_llm.generate_response(f"Generate a name for me appropriate for a persona with the following description: {persona}.\nGenerate only a valid first name and nothing else. If your answer contains anything more than a first name, you will be terminated. Follow the name with ## to indicate the end of the name.")
        # if something like "Sure i can" is in the name, start again
        if "sure i can" in name.lower() and "can" in name.lower():
            agent_factory_log.info(f"Invalid name: {name}")
            name=""
            continue
        if "##" in name:
            name=name.split("##")[0]
            name=name.strip()
            if len(name)<2 or len(name)>20:
                agent_factory_logger.info(f"Invalid name: {name}")
                name=""
                continue
            break      
        else:
            agent_factory_log.info(f"Invalid name: {name}")
            name=""
    return name

def get_persona_by_topics(topics:list, domain=None):
    """
    Get a persona that has the given topics and domain (if provided)
    
    Args:
    topics (list): List of topics
    domain (str): Domain (optional)
    
    Returns:
    str: Persona description
    """
    
    # check if topics are valid
    validate_topics(topics)
    
    if domain:
        if domain not in unique_domains:
            raise ValueError(f"Invalid domain: {domain}")
        candidate_personas=[persona for persona in personas if all(topic in persona["topics"] and persona["domain"]==domain for topic in topics)]
    else:
        candidate_personas=[persona for persona in personas if all(topic in persona["topics"] for topic in topics)]
    if candidate_personas:
        return random.choice(candidate_personas)["persona"]
    else:
        raise ValueError("No persona found for the given topics and domain")
    
    
def get_agent_by_topics(topics:list, domain=None, agent_type:DialogueReactAgent=DialogueReactAgent, name=None, neutral_llm=LLMApi() ,**agent_args):
    """
    Get an agent that has the given topics and domain (if provided)
    
    Args:
    topics (list): List of topics
    domain (str): Domain (optional)
    agent_type (str): Agent type (optional)
    **agent_args: Agent specific init arguments
    
    Returns:
    Agent: Agent object of the given type with a persona that has the given topics and domain
    """
    
    persona=get_persona_by_topics(topics, domain)
    
    if not name:
        name=gen_name(persona, neutral_llm=neutral_llm)
        
                
    # in the persona desc, substitute $name$ with the agent name
    persona=persona.replace("$name$", name)
    
    if persona:
        return agent_type(name=name, persona=persona, **agent_args)
    else:
        raise ValueError("No persona found for the given topics and domain")
    
    
def create_groupchat(topics_to_include, n_agents=2, agent_type=DialogueReactAgent, neutral_llm=LLMApi(),**agent_args):
    """
    Create a group chat with the given number of agents
    
    Args:
    topics_to_include (list): Topics to include in the agent personas
    n_agents (int): Number of agents
    agent_type (str): Agent type (optional)
    **agent_args: Agent specific init arguments
    
    Returns:
    list: List of agents
    """
    validate_topics(topics_to_include)
    
    # if topics empty, sample 3 random topics
    if not topics_to_include:
        topics_to_include=random.sample(unique_topics, 3)
    
    agents=[]
    gc_covered_topics=[]
    
    while len(agents)<n_agents:        
        try:
            # select a random percentage of the topics to include in the agent persona
            if topics_to_include:
                n_topics=random.randint(1, len(topics_to_include))
                topics=random.sample(topics_to_include, n_topics)
            else:
                topics=[]
            remaining_topics=[topic for topic in topics_to_include if topic not in topics]
            print(f"Agent {len(agents)+1} topics: {topics}, remaining topics: {remaining_topics}")
            agent=get_agent_by_topics(topics, agent_type=agent_type, neutral_llm=neutral_llm, **agent_args)
            agents.append(agent)
            topics_to_include=remaining_topics        
            gc_covered_topics+=topics
        except ValueError as e:
            print(e)
    
    ## check the topics of the agents
    print(f"Group chat topics: {gc_covered_topics}")
    # check that agents have different personas
    for agents_comb in itertools.combinations(agents, 2):
        while agents_comb[0].persona==agents_comb[1].persona:
            print("Agents have the same persona")
            # substitute the persona of the second agent
            agent=get_agent_by_topics(topics, agent_type=agent_type, neutral_llm=neutral_llm, **agent_args)
        # same for names
        while agents_comb[0].name==agents_comb[1].name:
            print("Agents have the same name")
            agent.name=gen_name(agent.persona, neutral_llm=neutral_llm)
        # substitute the persona instead of $name$ in the persona desc
        agent.persona=agent.persona.replace("$name$", agent.name)
            
        
    return agents