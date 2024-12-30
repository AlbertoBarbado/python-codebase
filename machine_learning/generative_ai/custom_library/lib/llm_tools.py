import torch
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer

def llm_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    max_new_tokens: int = 1000,
    temperature: float = 0.1,
    return_full_text: bool = False,
    ) -> str:
    """
    Calls the model with the given prompt and returns the response.

    NOTE: Uses HF Inference API

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call. Defaults to "claude-3-5-sonnet-20241022".

    Returns:
        str: The response from the language model.
    """
    dct_params = {
        'model': model,
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


class ModelInference:
    def __init__(
        self,
        model_name: str = "MiniLLM/MiniPLM-Qwen-500M",
        device: str = None
        ):
        """
        Initialize the model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model to load (e.g., 'gpt2').
            device (str, optional): Device to load the model on ('cpu', 'cuda', or None). Defaults to None.

        # Example usage:
        # model_inference = ModelInference("gpt2")
        # result = model_inference.llm_call("Once upon a time")
        # print(result)

        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def llm_call(
            self,
            prompt: str,
            system_prompt: str = "",
            max_length: int = 1000,
            temperature: float = 0.1,
            return_full_text: bool = False
            ):
        """
        Generate text based on a prompt using the model.

        Args:
            prompt (str): The input prompt for the model.
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature; higher values result in more random outputs.

        Returns:
            str: The generated text.
        """
        # Encode the input prompt
        if system_prompt != "":
            input_prompt = system_prompt + '\n\n' + prompt
        else:
            input_prompt = prompt
        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

        # Generate text using the model
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not return_full_text:
            response = response.split(input_prompt)[-1]

        # Decode and return the generated text
        return response

