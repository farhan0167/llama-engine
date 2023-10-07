from huggingface_hub import login
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaModel:
  def __init__(self, hf_token, llama_model_card):
    self.hf_token = hf_token
    self.llama_model_card = llama_model_card
    self.access_hf = self.huggingface_access()
    self.quantization_config = self.set_quantization_config()
    self.tokenizer = self.load_tokenizer()
    self.model = self.load_model()
  
  def huggingface_access(self):
    login(token=self.hf_token)
  
  def set_quantization_config(self):
    nf4_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16
    )
    return nf4_config
  
  def load_tokenizer(self):
    # Llama 2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.llama_model_card)
    return tokenizer

  def load_model(self):

    # Llama 2 Model
    model = AutoModelForCausalLM.from_pretrained(
        self.llama_model_card,
        trust_remote_code=True,
        quantization_config=self.quantization_config,
        device_map='auto',
    )
    return model