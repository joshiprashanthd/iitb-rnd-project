import json
class Response:
    def __init__(self, file_path: str) -> None:
        self.query_prefix = "Query: "
        self.response_prefix = "Answer: "
        self.gen_prompt_from_examples(file_path)
        
    def gen_prompt_from_examples(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.prompt = data['instruction']
    
    def make_query(self,query: str):
        query = f"""{self.prompt}

Answer the given query.
{self.query_prefix}{query}
{self.response_prefix}"""
        return query

    def __call__(self, llm, query: str, **kwargs) -> str:
        generation_query = self.make_query(query)
        response = llm.respond(generation_query, **kwargs)
        return response