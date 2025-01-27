import asyncio
import torch
import transformers
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HuggingFaceWrapperAPI:
  def __init__(
        self,
        model_name: str = "MiniLLM/MiniPLM-Qwen-500M",
        device: str = None
        ):
        """
        Initialize parameters.

        Args:
            model_name (str): Name of the pre-trained model to use (e.g., 'gpt2').
            device (str, optional): Device to load the model on ('cpu', 'cuda', or None). Defaults to None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

  def generate(
      self,
      prompt: str,
      system_prompt: str = "",
      max_new_tokens: int = 1000,
      temperature: float = 0.1,
      return_full_text: bool = False,
      ) -> str:
      """
      TODO
      """
      dct_params = {
          'model': self.model_name,
          'max_new_tokens': max_new_tokens,
          'temperature': temperature,
          'return_full_text': return_full_text
          }
      client = InferenceClient()
      if system_prompt != "":
          input_prompt = system_prompt + '\n\n' + prompt
      else:
          input_prompt = prompt
      response = client.text_generation(input_prompt, **dct_params)
      return response

  def load_model(self):
        return self.model


class HuggingFaceModelLoad:
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

  def generate(
      self,
      prompt: str,
      system_prompt: str = "",
      max_new_tokens: int = 1000,
      temperature: float = 0.1,
      return_full_text: bool = False,
      ):
      """
      TODO
      """
      model = self.load_model()
      pipeline = transformers.pipeline(
          "text-generation",
          model=model,
          tokenizer=self.tokenizer,
          use_cache=True,
          device_map=self.device,
          max_new_tokens=max_new_tokens,
          temperature=temperature,
          do_sample=True,
          truncation=True,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.eos_token_id,
      )
      self.pipeline = pipeline

      # Encode the input prompt
      '''
      # OLD VERSION
      if system_prompt != "":
          input_prompt = system_prompt + '\n\n' + prompt
      else:
          input_prompt = prompt
      # Generate output
      output_dict = pipeline(input_prompt)
      if return_full_text:
        output = output_dict[0]["generated_text"]
      else:
        output = output_dict[0]["generated_text"][len(input_prompt):]
      return output
      '''
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt},
      ]

      # Generate output
      output_dict = pipeline(messages)
      output = output_dict[0]['generated_text'][-1]['content']
      return output
  

class HuggingFaceModelLoadAsync:
    def __init__(
            self,
            model_name: str = "MiniLLM/MiniPLM-Qwen-500M",
            device: str = None,
            use_quantization=True
    ):
        """
        Initialize the HuggingFace model for text generation with optional quantization.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model setup with optional quantization
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

    def generate(
            self,
            prompt: str,
            system_prompt: str = "",
            max_new_tokens: int = 1000,
            temperature: float = 0.1,
            return_full_text: bool = False,
    ):
        """
        Synchronously generate text from the model.
        """
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map=self.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            truncation=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            trust_remote_code=True
        )
        self.pipeline = pipeline

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            output_dict = self.pipeline(messages)
            output = output_dict[0]['generated_text'][-1]['content']
            return output
        except Exception as e:
            return f"Error during synchronous generation: {e}"

    async def _generate_inference(self, pipeline, messages):
        """Run the model inference asynchronously in a separate thread."""
        return await asyncio.to_thread(pipeline, messages)

    async def generate_async(
            self,
            prompt: str,
            system_prompt: str = "",
            max_new_tokens: int = 1000,
            temperature: float = 0.1,
            return_full_text: bool = False,
    ):
        """
        Asynchronously generate text from the model.
        """
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map=self.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            truncation=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            trust_remote_code=True
        )
        self.pipeline = pipeline

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            output_dict = await self._generate_inference(self.pipeline, messages)
            output = output_dict[0]['generated_text'][-1]['content']
            return output
        except Exception as e:
            return f"Error during asynchronous generation: {e}"