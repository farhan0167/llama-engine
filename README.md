# High Level API for Running Llama Models 🦙 on Colab

Load open source large language models on Google Colab with 16.7% less code. The following models are supported:
1. Llama-2 7B and 13B, and its variants
2. [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha), and its variants
3. [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), and its variants

## Getting Started

On Google Colab's free version, select a T4 GPU Runtime, and run the following:
```
!pip install git+https://github.com/farhan0167/llama-engine.git
```

### Load any of 7B or 13B Llama Models
```python
from llama.llama_model import LlamaModel
from llama.chat import Chat

llama = LlamaModel(
    hf_token="insert your huggingface token here",
    llama_model_card="meta-llama/Llama-2-13b-chat-hf"
)
```
This should load the Llama-2-13B Chat model with 4 bit Quantization and should be enough to load the model on a free Colab GPU runtime with 16GB GPU RAM. 

### Run continuous Chat
```python
llm = Chat(tokenizer=llama.tokenizer, model=llama.model)

messages=[{"role": "system", "content": "You are a helpful assistant"}]

llm.start_chat(messages=messages)
```

Output:
```
Type your prompt or type End Chat to end the chat(case sensitive):  Google stocks plummets after the search engine lost visitors to Microsoft's Bing
 Based on the text you provided, I would classify the sentiment as negative. The use of the word "plummets" implies a significant and sudden drop in Google's stock price, which is typically associated with negative news. Additionally, the fact that Google lost visitors to Microsoft's Bing suggests that the company is facing increased competition and potentially losing market share, which could also contribute to a negative sentiment.
Type your prompt or type End Chat to end the chat(case sensitive):  What would you say the sentiment is for Microsoft based on this news?
 Based on the same text, I would classify the sentiment for Microsoft as positive. The fact that Microsoft's Bing is gaining visitors at Google's expense suggests that the company is making progress in the search engine market and potentially gaining market share. This could be seen as a positive development for Microsoft and could contribute to a positive sentiment.
Type your prompt or type End Chat to end the chat(case sensitive):  Let's say Meta is a competitor to Microsoft and competes with Google on ad revenue from the search engine. How do you think Meta's stock will be affected?
 Based on the text provided, I would classify the sentiment for Meta as neutral. While Google's loss of visitors to Microsoft's Bing could potentially lead to a decrease in ad revenue for both companies, it is not explicitly stated in the text that Meta is directly impacted by this news. Therefore, the sentiment for Meta's stock would likely be neutral, as there is no clear indication of how the news about Google and Microsoft would affect the company.
Type your prompt or type End Chat to end the chat(case sensitive):  Let's say in the past, bad news for Google always meant good news for Meta. How would you change your answer?
 Based on the additional information provided, I would classify the sentiment for Meta as positive. If in the past, bad news for Google has meant good news for Meta, then the news that Google's stock plummeted after losing visitors to Microsoft's Bing could be seen as positive for Meta. This is because Meta has historically benefited from Google's struggles, so the negative news for Google could be interpreted as a positive development for Meta. Therefore, the sentiment for Meta's stock would likely be positive.
Type your prompt or type End Chat to end the chat(case sensitive):  End Chat
```

### Save Chat History
You can save the history of the chat once you are done interacting by:
```python
llm.save_chat()
```
You will also be prompted to clear the chat history, which you should if you plan to start a new conversation. 

### Chat API
The Chat API provides a OpenAI like interface, and can be called as follows:
```python
response = llm.chat(
    user_prompt="What is the capital of Bangladesh?",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"}
    ], 
    show_prompt=False
    )

print(response['output'])
```

## Roadmap

- [ ] Introduce OpenAI GPT Function like capabilities
- [ ] Provide a low code framework to fine-tune LLMs using QLoRA, and the ability to attach/detach adapters