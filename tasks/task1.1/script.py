import sys, os
sys.path.append("/home/suraj/MedCoQ/hallucinations_in_LLMs")

from src.Model import Model
from src.configs import Llama38BConfig
from src.utils import generate_time_string, get_queries
from typing import List
import json
from tqdm import tqdm

LOG_DIR_PATH = "./logs"

def experiment(dataset_name: str, queries: List[str],  model: Model):
    os.makedirs(LOG_DIR_PATH, exist_ok=True)
    
    log = dict()
    log['model'] = model.config.name
    log['dataset_name'] = dataset_name
    log['time'] = generate_time_string()
    
    for i, query in tqdm(queries, ncols=100):
        # print(f"{i}. {query}")
        answer = model.gen_answer(query) # get original response
        log[i] = {"query": query, 'answer': answer}
    
        with open(os.path.join(LOG_DIR_PATH, f"{dataset_name}_{log['model']}_{log['time']}.json"), 'w') as file:
            file.write(json.dumps(log))
            
    return log

def pubmed_final_exp():
    response_json_path = "/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/pubmed_without_reasoning_response.json"
    csv_name = "pubmed_final"
    csv_path = f"/home/suraj/MedCoQ/dataset/factchecking/{csv_name}.csv"
    model = Model(verbose=False, response_json_path=response_json_path)
    queries = get_queries(csv_path, max_len=100)
    experiment(csv_name, queries, model)

def med_coq_exp():
    response_json_path = '/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/response_without_reasoning.json'
    csv_name = "medcoq_sample"
    csv_path = f"/home/suraj/MedCoQ/dataset/factchecking/{csv_name}.csv"
    model = Model(verbose=False, response_json_path=response_json_path)
    queries = get_queries(csv_path, max_len=100)
    experiment(csv_name, queries, model)
    
def bio_asq_exp():
    response_json_path = '/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/response_without_reasoning.json'
    csv_name = "bioasq_facts"
    csv_path = f"/home/suraj/MedCoQ/dataset/factchecking/{csv_name}.csv"
    model = Model(verbose=False, response_json_path=response_json_path)
    queries = get_queries(csv_path, max_len=100)
    experiment(csv_name, queries, model)
    
if __name__ == '__main__':
    print("RUNNING MEDCOQ EXPERIMENT")
    med_coq_exp()
    
    print("RUNNING BIOASQ EXPERIMENT")
    bio_asq_exp()