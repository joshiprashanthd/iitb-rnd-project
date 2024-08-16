class Config:
    def __init__(self, name: str, path: str, prompt_template: str, instruction_end_token: str) -> None:
        self.name = name
        self.path = path
        self.prompt_template = prompt_template
        self.instruction_end_token = instruction_end_token
        
    def extract_response(self, output):
        last_output_translation: str = output['choices'][-1]['text']
        inst_end_index = last_output_translation.rfind(self.instruction_end_token)
        response = last_output_translation[inst_end_index + len(self.instruction_end_token):].strip()
        return response
    

class MixtralConfig(Config):
    def __init__(self) -> None:
        super().__init__("Mixtral8x7B", "/home/suraj/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", '''<s> [INST] {text} [/INST]''', "[/INST]")
    

class Phi2Config(Config):
    def __init__(self) -> None:
        template = '''Instruct: {text}
Output:'''

        super().__init__("Phi2B", "/home/suraj/MedCoQ/hallucinations_in_LLMs/models/phi-2.Q8_0.gguf", template, 'Output:')
        
class Llama38BConfig(Config):
    def __init__(self) -> None:
        template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        super().__init__("LLaMA38B", "/home/suraj/MedCoQ/hallucinations_in_LLMs/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", template, "<|end_header_id|>")