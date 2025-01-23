import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable

from lib.utils import extract_xml

def chain(input: str, prompts: List[str], model: object, dct_params: dict, debug_mode=True) -> str:
    """Chain multiple LLM calls sequentially, passing results between steps."""
    PROMPT_TEMPLATE = "{PROMPT}\nInput: {RESULT}"
    result = input
    for i, prompt in enumerate(prompts, 1):
        print(f"\nStep {i}:")
        #result = model.generate(f"{prompt}\nInput: {result}", **dct_params)
        result = model.generate(
            PROMPT_TEMPLATE.replace("{PROMPT}", prompt).replace("{RESULT}", result),
            **dct_params
            )
        print(result)
        if debug_mode:
          time.sleep(10)
    return result

def parallel(prompt: str, inputs: List[str], model: object, dct_params: dict, n_workers: int = 3) -> List[str]:
    """Process multiple inputs concurrently with the same prompt."""
    PROMPT_TEMPLATE = "{PROMPT}\nInput: {RESULT}"
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        #futures = [executor.submit(model.generate, f"{prompt}\nInput: {x}", **dct_params) for x in inputs]
        #replacements = str.maketrans({"{PROMPT}": prompt, "{RESULT}": x})
        #futures = [executor.submit(model.generate, PROMPT_TEMPLATE.translate(replacements), **dct_params) for x in inputs]
        futures = [
            executor.submit(
                model.generate, PROMPT_TEMPLATE.replace("{PROMPT}", prompt).replace("{RESULT}", x),
                **dct_params
                )
            for x in inputs
            ]
        return [f.result() for f in futures]

def route(input: str, routes: Dict[str, str], model: object, dct_params: dict) -> str:
    """Route input to specialized prompt using content classification."""
    # First determine appropriate route using LLM with chain-of-thought
    print(f"\nAvailable routes: {list(routes.keys())}")
    SELECTOR_PROMPT = f"""
    Analyze the input and select the most appropriate support team from these options: {list(routes.keys())}
    First explain your reasoning, then provide your selection in this XML format:

    <reasoning>
    Brief explanation of why this ticket should be routed to a specific team.
    Consider key terms, user intent, and urgency level.
    </reasoning>

    <selection>
    The chosen team name
    </selection>

    Input: {input}""".strip()
    PROMPT_TEMPLATE = "{PROMPT}\nInput: {RESULT}"

    route_response = model.generate(SELECTOR_PROMPT, **dct_params)
    reasoning = extract_xml(route_response, 'reasoning')
    route_key = extract_xml(route_response, 'selection').strip().lower()

    print("Routing Analysis:")
    print(reasoning)
    print(f"\nSelected route: {route_key}")

    # Process input with selected specialized prompt
    selected_prompt = routes[route_key]
    return model.generate(PROMPT_TEMPLATE.replace("{PROMPT}", selected_prompt).replace("{RESULT}", input), **dct_params)
    #return model.generate(f"{selected_prompt}\nInput: {input}", **dct_params)