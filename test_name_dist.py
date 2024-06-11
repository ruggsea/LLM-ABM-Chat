from agent_factory import gen_name, read_personas
import json, random


personas=read_personas()

# select 100 random personas, and generate 1000 names for each persona, saving the names in a file as a counter

personas=random.sample(personas, 50)

personas=[persona["persona"] for persona in personas]

file="names_dist.jsonl"

for persona in personas:
    print(f"Generating names for persona: {persona}")
    names_counter={}
    for i in range(100):
        name=gen_name(persona)
        if name not in names_counter:
            names_counter[name]=0
        names_counter[name]+=1
    # sort counter by biggest counts
    names_counter={k: v for k, v in sorted(names_counter.items(), key=lambda item: item[1], reverse=True)}
    print(f"Names generated: {names_counter}")
        
    with open(file, "a") as f:
        f.write(json.dumps({"persona":persona,"names_counter":names_counter})+"\n")