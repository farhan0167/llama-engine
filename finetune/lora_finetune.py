import os
import torch
from typing import Optional, Union, List
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from finetune.finetune import Finetune

class LoraFinetune(Finetune):
    def __init__(self, model, dataset) -> None:
        super().__init__(model, dataset)
        self.lora_config = self.set_lora_config()
    
    def set_lora_config(
            self,
            lora_alpha:int = 16,
            lora_dropout:float = 0.1,
            r:int = 64,
            bias:str = "none",
            task_type:str = "CAUSAL_LM",
            target_modules: Optional[Union[List[str], str]] = None
    ): 
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=r,
            bias=bias,
            task_type=task_type,
            target_modules=target_modules
        )
        return peft_config
    
    def trainer(
            self,
            dataset_text_field:str = "output",
            max_seq_length:int | None = None,
            packing:bool = False
    ):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            dataset_text_field=dataset_text_field,
            peft_config=self.lora_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=packing,
        )
        return trainer
        
