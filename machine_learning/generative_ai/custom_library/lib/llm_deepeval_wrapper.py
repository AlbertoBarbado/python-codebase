import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

class HuggingFaceWrapperDeepEval(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "MiniLLM/MiniPLM-Qwen-500M",
        device: str = None,
        use_quantization = True
        ):
        """
        TODO
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

        # Model setup
        if use_quantization:
          quantization_config = BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_compute_dtype=torch.float16,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_use_double_quant=True,
          )
          self.model = AutoModelForCausalLM.from_pretrained(
              self.model_name,
              device_map=self.device,
              quantization_config=quantization_config,
          )
        else:
          self.model = AutoModelForCausalLM.from_pretrained(
              self.model_name,
              trust_remote_code=True
              )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        #parser = JsonSchemaParser(schema.schema())
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name