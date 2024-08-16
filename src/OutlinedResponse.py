import json
class OutlinedResponse:
    def __init__(self, file_path: str) -> None:
        self.outline_prefix = "Outline: "
        self.query_prefix = "Query: "
        self.response_prefix = "Answer: "
        self.gen_prompt_from_examples(file_path)
        
    def gen_prompt_from_examples(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        self.prompt = data['instruction']
    
    def make_query(self, outline: str, query: str):
        query = f"""{self.prompt}

{self.outline_prefix}
{outline}
{self.query_prefix}{query}
{self.response_prefix}"""
        return query

    def __call__(self, llm, outline: str, query: str, **kwargs) -> str:
        generation_query = self.make_query(outline, query)
        response = llm.respond(generation_query, **kwargs)
        return response