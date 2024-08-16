from typing import List, Union
from .utils import clean_strings, extract_multiline_points

from llama_cpp import Llama
from .configs import Config, MixtralConfig
from .OutlinedResponse import OutlinedResponse
from .Response import Response
from .Points import Points
from .Outline import Outline

class Model:
    def __init__(self, 
                 config: Config = MixtralConfig(), 
                 verbose=False, 
                 points_json_path: str = "/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/points.json", 
                 outline_json_path: str = "/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/outline.json",
                 response_json_path: str = "/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/response_without_reasoning.json",
                 outlined_response_json_path: str = "/home/suraj/MedCoQ/hallucinations_in_LLMs/data/misc/outlined_response.json"):
        
        self.config = config
        self.llm = Llama(
            model_path=self.config.path,
            n_ctx=5000,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=35,  # The number of layers to offload to GPU, if you have GPU acceleration available
            n_batch=512,
            verbose=verbose,
        )
        
        self.outline_generator = Outline(outline_json_path)
        self.points_generator = Points(points_json_path)
        self.response_generator = Response(response_json_path)
        self.outlined_response_generator = OutlinedResponse(outlined_response_json_path)
    
    def respond(self, prompt, stop: Union[str, List[str]] = "</s>", max_tokens: int = 512, temperature: float = 0.8) -> str:
        my_prompt = self.config.prompt_template.format(text=prompt)
        output = self.llm(
            my_prompt,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=True,
        )
        return self.config.extract_response(output)

    def gen_outline(self, query: str):
        return self.outline_generator(self, query)
    
    def gen_outlined_answer(self, outline: str, query: str, stop: Union[str, List[str]] = "</s>", max_tokens: int = 512, temperature: float = 0.8):
        return self.outlined_response_generator(self, outline, query, stop=stop, max_tokens=max_tokens, temperature=temperature)
    
    def gen_outlined_points(self, text: str):
        return extract_multiline_points(text)

    def gen_answer(self, query: str, stop: Union[str, List[str]] = "</s>", max_tokens: int = 512, temperature: float = 0.8):
        return self.response_generator(self, query, stop=stop, max_tokens=max_tokens, temperature=temperature)
        
    def gen_points(self, text: str):
        return self.points_generator(self, text)
    
    def gen_refined_response(self, refined_points: List[str]):
        return " ".join(clean_strings(refined_points)).strip()
      