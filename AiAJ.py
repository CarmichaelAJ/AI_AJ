import time
import os
import timeit
from dotenv import load_dotenv
from langchain.llms.base import LLM
from llama_index import ( PromptHelper, SimpleDirectoryReader, LLMPredictor,ServiceContext, GPTListIndex)
from transformers import pipeline
##from GPTListIndex import GPTListIndex
import torch

load_dotenv()

def timeit():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start=time.time()
            result= func(*args, **kwargs)
            end = time.time()
            args=[str(arg) for arg in args]

            print(f"[{(end-start):.8f} seconds]: f({args}) -> {result}")
            return result
        return wrapper
    return decorator


max_token = 512

prompt_helper = PromptHelper(
    max_input_size=1024,
    num_output=256,
    max_chunk_overlap=20
)

class localOPT(LLM):
    model_name = "facebook/opt-iml-max-1.3b" ## Get a small model lol 30b is 60gb
    pipeline = pipeline("text-generation", model=model_name, model_kwarfs={"torch_dtype":torch.bfloat16 })
    
    def _call(self, prompt:str, stop=None)-> str:
        response = self.pipeline(prompt, max_new_tokens=max_token)[0]["generated_text"]
        ## Only returning newly generated tokens w/out prompt
        return response[len(prompt):]

    @property
    def _indentifying_params(self):
        return{"name_of_model": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"

@timeit()
def create_index():
    print("Creating Index...")
    llm=LLMPredictor(llm= localOPT())
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm,
        prompt_helper=prompt_helper
    )
    docs = SimpleDirectoryReader('squad').load_data()
    index = GPTListIndex.from_documents(docs, service_context=service_context)
    print("Done")
    return index

@timeit()
def execute_query():
    response = index.query("What did the president say?"
        #exclude_keywords=[""],
        #required_keywords[""],
        )
    ##response_mode="no_text"
    return response

if __name__ == "__main__":

    filename = "AiAJ.json"
    if not os.path.exists(filename):
        print("No local cache of the model, downloading right now...")
        index = create_index()
        index.save_to_disk(filename)
    else: 
        print("Loading local cache of the embeddings")
        ##index = GPTListIndex.load_from_disk(filename)
        
    response = execute_query()
    print(response)
    ##print(response.source_nodes())
    ##print(response.get_formatted_sources())
