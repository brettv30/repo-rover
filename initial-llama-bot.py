from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

prompt = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search

You have access to the following functions:

Use the function 'spotify_trending_songs' to: Get top trending songs on Spotify
{{
  "name": "spotify_trending_songs",
  "description": "Get top trending songs on Spotify",
  "parameters": {{
    "n": {{
      "param_type": "int",
      "description": "Number of trending songs to get",
      "required": true
    }}
  }}
}}

If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line"
- Always add your sources when using search results to answer the user query

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION: {question} 
 <|eot_id|><|start_header_id|>assistant<|end_header_id|>                    
    """
)


llm = OllamaFunctions(
    model="llama3.1:8b-instruct-q2_K",
    temperature=0,
    num_gpu=1,
    verbose=True,
)

chain = prompt | llm | JsonOutputParser()

response = chain.invoke(
    {"question": "Can you check the top 5 trending songs on spotify?"}
)

print(response)

# for chunk in chain.stream(
#     {"question": "Can you help me solve this equation: x^3 - 4x^2 + 6x - 24 = 0"}
# ):
#     print(chunk, end="", flush=True)
