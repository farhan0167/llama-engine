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
  
  def print_trainable_parameters(self):
    """
    Prints the number of trainable parameters in the model.
    Adapted from https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k#scrollTo=gkIcwsSU01EB
    """
    if not self.model:
      return None
    
    trainable_params = 0
    all_param = 0
    for _, param in self.model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {
      "trainable_params": trainable_params,
      "all_param": all_param,
      "trainable %": 100 * trainable_params / all_param
    }
  
  def estimate_memory_footprint(self):
    """
    Math derived from EleutherAI Transformer Math 101: https://blog.eleuther.ai/transformer-math/
    """
    if not self.model:
      return None
    
    model_params = self.print_trainable_parameters()
    total_params = model_params.get("all_param")
    marker = "="*15
    byte = 8
    int4 = 4
    int8 = 8
    bf16_fp16 = 16
    fp32 = 32
    gb_unit = 1024**3

    fp32_memory = ((fp32/byte)*total_params)/gb_unit
    bf16_fp16_memory = ((bf16_fp16/byte)*total_params)/gb_unit
    int8_memory = ((int8/byte)*total_params)/gb_unit
    int4_memory = ((int4/byte)*total_params)/gb_unit

    inference_fp32_memory = fp32_memory * 1.2
    inference_bf16_fp16_memory = bf16_fp16_memory * 1.2
    inference_int8_memory = int8_memory * 1.2
    inference_int4_memory = int4_memory * 1.2

    print(marker)
    print("Inference Memory Requirement (Approximation)")
    print(f"Floating Point 32 {inference_fp32_memory} GB")
    print(f"Floating Point 16 {inference_bf16_fp16_memory} GB")
    print(f"Int 8 {inference_int8_memory} GB")
    print(f"Int 4 {inference_int4_memory} GB")
    print(marker)




