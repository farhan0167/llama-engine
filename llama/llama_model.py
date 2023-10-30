from huggingface_hub import login
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaModel:
  def __init__(
      self, 
      hf_token:str | None = None, 
      llama_model_card:str = "meta-llama/Llama-2-13b-chat-hf",
      load_model:bool = True,
      quantize_model:bool = True,
      trainable:bool = False
  ):
    self.hf_token = hf_token
    self.llama_model_card = llama_model_card
    self.access_hf = self.huggingface_access()
    self.quantization_config = self.set_quantization_config() if quantize_model else None
    self.quantization_config = self.set_quantization_config(bnb_4bit_use_double_quant=False) if trainable else self.quantization_config
    self.quantization_config = self.quantization_config if quantize_model else None #note: might need refactoring. this is for if trainable but not quantized.
    self.tokenizer = self.load_tokenizer()
    self.model = self.load_model() if load_model else None
  
  def huggingface_access(self):
    if self.hf_token:
      try:
        login(token=self.hf_token)
      except Exception as e:
        print("Error logging in to HuggingFace: ", e)
        return None

  
  def set_quantization_config(
      self,
      load_in_4bit: bool = True,
      bnb_4bit_quant_type: str = "nf4",
      bnb_4bit_use_double_quant:bool = True,
      bnb_4bit_compute_dtype = torch.bfloat16
    ):
    nf4_config = BitsAndBytesConfig(
      load_in_4bit=load_in_4bit,
      bnb_4bit_quant_type=bnb_4bit_quant_type,
      bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
      bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
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