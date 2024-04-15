import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def get_files():
    files = []
    for root, dirs, file in os.walk("MPC CORPUS/chat_sessions_annotated"):
        for f in file:
            files.append(os.path.join(root, f))
    return files

def read_file(file):
    tree = ET.parse(file)
    convo = file.replace(".xml","")
    root = tree.getroot()
    messages = []
    for message in root.findall('turn'):
        message_dict = {}
        for key, value in message.items():
            message_dict[key] = value
        message_dict['text'] = message.text
        message_dict['session'] = convo
        messages.append(message_dict)
    return messages

def main():
    # get into the MPC CORPUS directory
    os.chdir("fewshots_mpc/")
    print("Getting files")
    files = get_files()
    messages = []
    for file in files:
        messages += read_file(file)

    df = pd.DataFrame(messages)



    print("Filtering out empty dialog acts")
    df = df[df.dialog_act!=""]
    #
    # take out prefix from dialog act
    df['dialog_act'] = df['dialog_act'].apply(lambda x: x.split(":")[1])
    
    # keep only text, speaker and dialog_act
    df = df[['text', 'speaker', 'dialog_act']]
    
    
    print("Saving to jsonl")
    df.to_json("mpc_messages.jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()

