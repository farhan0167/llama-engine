"""
Code adapted from https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.Trainer, and
https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing
"""

import os
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from llama.llama_model import LlamaModel

class Finetune:
    def __init__(
            self,
            model: LlamaModel,
            dataset: object
    ) -> None:
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.dataset = dataset
        self.training_args = self.set_training_arguments()

    def set_training_arguments(
            self,
            output_dir:str="./results",
            num_train_epochs:int=1,
            per_device_train_batch_size:int=4,
            gradient_accumulation_steps:int=1,
            optim:str="paged_adamw_32bit",
            save_steps:int=0,
            logging_steps:int=25,
            learning_rate:float=2e-4,
            weight_decay:float=0.001,
            fp16:bool=False,
            bf16:bool=False,
            max_grad_norm:float=0.3,
            max_steps:int=-1,
            warmup_ratio:float=0.03,
            group_by_length:bool=True,
            lr_scheduler_type:str="cosine",
            logging_dir:str = "./logs"
    ):
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            logging_dir=logging_dir
        )
        return training_arguments
    
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
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=packing,
        )
        return trainer
    
    def finetune(
            self,
            dataset_text_field:str = "output",
            max_seq_length:int | None = None,
            packing:bool = False,
            save_model:bool = True,
            new_model_name:str = "finetuned_model_card",
            train_epochs:int = 1,
            batch_size: int= 4,
            max_steps:int = -1
    ):
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.training_args = self.set_training_arguments(
            num_train_epochs=train_epochs, per_device_train_batch_size=batch_size, max_steps=max_steps
        )
        trainer = self.trainer(
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length
        )
        trainer.train()

        if save_model:
            trainer.model.save_pretrained(new_model_name)

        return trainer


