from transformers import pipeline
import json

class Chat:
  def __init__(self, tokenizer, model):
    self.tokenizer = tokenizer
    self.model = model
    self.hf_pipline = self.load_generator()
    self.messages = [
        {"role":"system", "content": ""},
    ]

  def save_chat(self, fname:str="example.json"):
    messages = self.messages
    to_json = json.dumps(messages)
    with open(fname, "w") as outfile:
      outfile.write(to_json)
    
    to_reset_chat = input("Do you want to clear your chat history? Type Y/N/y/n: ")

    while to_reset_chat.lower() not in ["y","n"]:
      to_reset_chat = input("Invalid Input. Try again. Do you want to clear your chat history? Type Y/N/y/n: ")

    if to_reset_chat.lower() == "y":
      self.messages = [
        {"role":"system", "content": ""},
        ]
      return "Chat History Saved to File and Cleared"
    elif to_reset_chat.lower() == "n":
      return "Chat History Saved to File"
  
  def load_generator(self, temp=0.1, max_new_tokens=500, repetition_penalty=1.1):
    # Our text generator
    generator = pipeline(
        model=self.model, tokenizer=self.tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    return generator
  
  def parse_message(self, messages:list):
    system_prompt = "<s>[INST] <<SYS>>\n" + messages[0]['content'] + "\n<</SYS>>\n"
    if len(messages) == 2:
      #system_prompt = "<s>[INST] <<SYS>>\n" + messages[0]['content'] + "\n<</SYS>>"
      user_prompt = messages[1]['content'] + "[/INST]"
      prompt = system_prompt + user_prompt
      return prompt
    else:
      prompt = system_prompt
      for i in range(1,len(messages)):
        if i%2 ==0:
          prompt+= "<<SYS>>\n" + messages[i]['content'] + "\n<</SYS>>\n"
        else:
          prompt+=messages[i]['content'] + "\n"
      prompt+="[/INST]"
      return prompt


  def format_output(self,output):
    l, r = 0, len("[/INST]")
    while r<len(output):
      if "[/INST]" == output[l:r]:
        return output[r+1:]
        break
      else:
        l +=1
        r+=1

  def chat(self, user_prompt: str, messages:list, show_prompt: bool):
    self.messages.append({"role":"user", "content": user_prompt})
    prompt = self.parse_message(self.messages)
    if show_prompt:
      print("Prompt: ", prompt)
    sequences = self.hf_pipline(prompt)
    output = sequences[0]['generated_text']
    formatted = self.format_output(output)
    self.messages.append({"role":"system", "content": formatted})

    response = {
        "output": formatted,
        "messages": self.messages
    }
    return response
  
  def start_chat(self, sys_prompt=None):
    if not sys_prompt:
      sys_prompt = """
      You are a helpful assistant who answers questions as truthfully as possible. If you dont know an answer, say I dont know.
      """
    self.messages[0]["content"] = sys_prompt
    while True:
      user_prompt = input("Type your prompt or type End Chat to end the chat(case sensitive):  ")
      if user_prompt == "End Chat":
        break
      response = self.chat(user_prompt=user_prompt, messages=self.messages, show_prompt=False)
      print(response['output'])
      self.messages = response['messages']