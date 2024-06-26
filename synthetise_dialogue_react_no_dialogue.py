# loading the messages from fewshot_mpc/mpc_messages.jsonl, use them to generate dialogue-react examples
# and then saving the generated conversations in a jsonl file


# import llm library 
from llm_engines import ChatgptLLM
from jinja2 import Template
import json, random, re, logging
import numpy as np


# setup prompt logging in synthetis.log
logging.basicConfig(filename='synthetis.log', level=logging.INFO)



# test the llm model
def generate_examples(llm, template, pre_examples, messages):
    # messages should be a list, each element is a message
    
    context = {
        "messages": messages,
        "pre_examples": pre_examples
        }
    
    template = template.render(context)
    logging.info(f"Rendered prompt: {template}")
    response = llm.generate_response(template)
    
    return response
    
    
        
        
        
def main():
    # load the llm model
    llm = ChatgptLLM()
    llm.model= "gpt-4-turbo-preview"
    
    # load the jinja template
    with open('fewshots_mpc/mpc_react.j2', 'r') as f:
        template = Template(f.read())
    # load the messages
    messages_path= "fewshots_mpc/mpc_messages.jsonl"
    # load jsonl
    with open(messages_path, "r") as f:
        messages= [json.loads(line) for line in f]
    print(f"Total messages: {len(messages)}")
    
    
    
    ## loading pre_examples
    ## pre_examples are contained in the pre_examples.txt file, they are delimited by a line with "### pre_example n"
    pre_examples_path= "fewshots_mpc/pre_examples_no_dialogue.txt"
    with open(pre_examples_path, "r") as f:
        pre_examples= f.readlines()
    
    pre_examples= [pre_example.strip() for pre_example in pre_examples]
    # make it one string
    pre_examples= "\n".join(pre_examples)
    ## split the pre_examples by "### pre_example n"
    dividing_pattern= re.compile(r"### pre_examples \d+")
    pre_examples= dividing_pattern.split(pre_examples)
    pre_examples= [pre_example.strip() for pre_example in pre_examples]
    # drop pre_examples that are empty
    pre_examples= [pre_example for pre_example in pre_examples if pre_example]   
    print(f"Total pre_examples: {len(pre_examples)}")
    
    
    file= open("fewshots_mpc/generated_dialogue_reacts_no_dialogue.jsonl", "a")
    n_iterations= 100
    
    print(f"Generating {n_iterations} times")
    
    # update i to be the latest id in the file if it exists
    try:
        with open("fewshots_mpc/generated_dialogue_reacts.jsonl", "r") as f:
            lines= f.readlines()
            last_line= lines[-1]
            last_id= json.loads(last_line)["id"]
            starting_file_index= last_id+1
    except:
        starting_file_index= 0

    # sample a uni
    for i in range(n_iterations):
        
        i= i+starting_file_index
            
        n_messages= int(random.normalvariate(15, 2))
        
        if n_messages < 7:
            n_messages= 7
            
        if n_messages > 30:
            n_messages= 30
        print("Generation number: ", i)
        print(f"Generating a conversation with {n_messages} messages")
        
        starting_index= random.randint(0, len(messages)-n_messages)
        messages_for_gen= messages[starting_index:starting_index+n_messages]
        
        # get dialogue acts out of the message dict, the dict should have the keys "text" and "speaker"
        messages_for_gen= [{"text": message["text"], "speaker": message["speaker"]} for message in messages_for_gen]        
        print(f"Starting index: {starting_index}")
        
        pre_examples_for_gen= random.sample(pre_examples, 1)
        
        response= generate_examples(llm, template, pre_examples_for_gen, messages_for_gen)

        print(f"Generated response of length: {len(response)}")
        conv_dict= {
            "id": i,
            "pre_examples": pre_examples_for_gen,
            "messages": messages_for_gen,
            "generated_response": response
            }
        dump= json.dumps(conv_dict)
    
        file.write(dump+"\n")
    
    
    
if __name__ == "__main__":
    main()