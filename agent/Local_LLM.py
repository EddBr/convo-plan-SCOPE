"""Contains classes for querying local large language models."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from agent.LLM import LLM, ICL_prompt

from huggingface_hub import login

cache_dir = "~/.cache/huggingface"  # Or shared project space


class Local_LLM(LLM):
    def __init__(self, model_config, model = None, tokenizer = None):
        # self.model_name = model_config["name"]
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_config"]["pretrained_model_name_or_path"], cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        if model is None:
            print(f'Loading model {model_config["model_config"]["pretrained_model_name_or_path"]} on device {model_config["model_config"]["device_map"]} in Local_LLM...')
            self.model = AutoModelForCausalLM.from_pretrained(torch_dtype=torch.bfloat16, **model_config["model_config"], cache_dir=cache_dir)
        else:
            self.model = model
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.tokenizer.apply_chat_template([{"role":"system","content":""}])
            self.tokenizer_has_system_prompt = True
        except:
            self.tokenizer_has_system_prompt = False

        # Convert generation_config dict to GenerationConfig object to avoid attribute errors
        gen_config_dict = model_config["generation_config"].copy()
        # Fix: If num_beam_groups equals num_beams, remove it to use regular beam search
        # This avoids the 'transformers-community/group-beam-search' AttributeError
        if gen_config_dict.get("num_beam_groups") == gen_config_dict.get("num_beams"):
            gen_config_dict.pop("num_beam_groups", None)
        self.generation_config = GenerationConfig(**gen_config_dict)
        self.system_prompt = ICL_prompt(model_config)

    def generate(self, chat : list[dict], **kwargs) -> list[str]:
        tokens = self.tokenizer.apply_chat_template(
            chat, 
            tokenize = True, add_generation_prompt = True, return_tensors = "pt", return_attention_mask = True, return_dict = True
            ).to(self.model.device)
        with torch.no_grad():
            # Update generation_config with any kwargs that are generation parameters
            # This avoids issues with unpacking dicts that might contain invalid generation modes
            # Convert to dict, update, then create new GenerationConfig
            config_dict = self.generation_config.to_dict()
            other_kwargs = {}
            for key, value in kwargs.items():
                if key in config_dict:
                    config_dict[key] = value
                else:
                    other_kwargs[key] = value
            
            # Root fix: Remove num_beam_groups if it equals num_beams (not actually diverse beam search)
            # This prevents the 'transformers-community/group-beam-search' AttributeError
            if config_dict.get("num_beam_groups") == config_dict.get("num_beams"):
                config_dict.pop("num_beam_groups", None)
            
            updated_config = GenerationConfig(**config_dict)
            output = self.model.generate(
                input_ids=tokens["input_ids"], 
                attention_mask=tokens["attention_mask"], 
                generation_config=updated_config,
                **other_kwargs
            )
        output = output[:, tokens["input_ids"].shape[-1]:]    # Only return generated tokens
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        decoded_output = (list(set(decoded_output))) # remove duplicates
        print("generated LLM output after removing duplicate: ", len(decoded_output))
        return decoded_output

