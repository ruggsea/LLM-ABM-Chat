## loading fits data, getting topics

import os
import pandas as pd
import json


def get_files_in_folder(relative_path):
    # cwd
    current_path = os.getcwd()
    absolute_path = current_path + relative_path
    # get all folders in the relative path and the files inside them
    files=[]
    for folder in os.listdir(absolute_path):
        # if path is a folder
        if os.path.isdir(os.path.join(absolute_path, folder)):
            folder_path = os.path.join(absolute_path, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.txt'):
                    files.append(os.path.join(folder_path, file))
    
    return files



def read_files(files):
    # read all files
    data = []
    for file in files:
        with open(file, 'r') as f:
            # each line is a json
            for line in f:
                line_dict=json.loads(line)
                data.append(line_dict)
    
    return data




def main():
    # cd to the folder with the data - fits_personas
    os.chdir('fits_personas')

    
    relative_path = '/fits'
    print("Getting files in folder")
    files = get_files_in_folder(relative_path)
    data = read_files(files)
    df = pd.DataFrame(data)
    print("Extracting topics")
    df_topics = df[['generic_topic', 'domain']].drop_duplicates()
    # save the topics to a json file
    print("Saving topics to json")
    df_topics.to_json('topics.json', orient='records')
    
    
if __name__ == '__main__':
    main()