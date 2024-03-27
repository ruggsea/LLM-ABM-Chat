# loading the topics from topics.json, using a jinja template to generate the persona using the llm model
# and then saving the generated personas in a jsonl file


# import llm library 
from llm_engines import ChatgptLLM
from jinja2 import Template
import json, random
import numpy as np



# test the llm model
def generate_persona(llm, template, topics, domains):
    
    # generate persona using llm
    context = {
        "topics": topics,
        "domains": domains,
    }
    # render the template
    persona_template = template.render(context)
    # generate response
    response = llm.generate_response(persona_template)

    return response

def main():
    # load the llm model
    llm = ChatgptLLM()
    # load the jinja template
    with open('fits_personas/example_persona_template.j2', 'r') as f:
        template = Template(f.read())
    # load the topics
    with open('fits_personas/topics.json', 'r') as f:
        topics = json.load(f)
    
    # topics is a list of json objects with keys: domain, generic_topic
    domains = [topic["domain"] for topic in topics]
    topics = [topic["generic_topic"] for topic in topics]

    # Remove duplicates from topics list
    topics = list(set(topics))
    domains = list(set(domains))

    print(f"Total topics: {len(topics)}")
    print(f"Total domains: {len(domains)}")
    
    system_prompt = "You are a very creative writer who specializes in descriptions of people. You tend to describe people vividly more based on their actions and behaviors than by their demographics. Your writing style is very simple and doesn't need complicated words, but you never sound repetitive or too verbose."
    llm.set_system_prompt(system_prompt)
    n_personas = 1000
    
    # set llm model to gpt4
    llm.model= "gpt-4-turbo-preview"
    
    file= open("fits_personas/generated_personas.jsonl", "a")
    
    # generate n personas
    # for each persona, sample 1-3 topics and 1 domain
    for i in range(n_personas):
        # Generate a random integer from a normal distribution
        mean = 3
        std = 1
        persona_topics_count = int(np.random.normal(mean, std))
        persona_topics = random.sample(topics, persona_topics_count)
        persona_domain = random.choice(domains)
        response = generate_persona(llm, template, persona_topics, persona_domain)
        print(f"Persona {i+1} (topics: {persona_topics}, domain: {persona_domain}):")
        print(response)
        print("\n\n")
        # append persona to file
        file.write(json.dumps({"persona": response,"topics": persona_topics, "domain":persona_domain}) + "\n")
    
    
if __name__ == "__main__":
    main()