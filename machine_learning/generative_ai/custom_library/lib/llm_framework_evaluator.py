import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable
from lib.utils import extract_xml
from lib.llm_tools import llm_call

def generate(prompt: str, task: str, dct_params: dict, context: str = "") -> tuple[str, str]:
    """Generate and improve a solution based on feedback."""
    full_prompt = f"{prompt}\n{context}\nTask: {task}" if context else f"{prompt}\nTask: {task}"
    print("\n=== INPUT START ===")
    print(f"Full prompt:\n{full_prompt}\n")
    print("\n=== INPUT END ===")

    response = llm_call(full_prompt, **dct_params)
    thoughts = extract_xml(response, "thoughts")
    result = extract_xml(response, "response")

    print("\n=== GENERATION START ===")
    print(f"Thoughts:\n{thoughts}\n")
    print(f"Generated:\n{result}")
    print("=== GENERATION END ===\n")

    return thoughts, result

def evaluate(prompt: str, content: str, task: str, dct_params: dict) -> tuple[str, str]:
    """Evaluate if a solution meets requirements."""
    full_prompt = f"{prompt}\nOriginal task: {task}\nContent to evaluate: {content}"
    response = llm_call(full_prompt, **dct_params)
    evaluation = extract_xml(response, "evaluation")
    feedback = extract_xml(response, "feedback")

    print("=== EVALUATION START ===")
    print(f"Status: {evaluation}")
    print(f"Feedback: {feedback}")
    print("=== EVALUATION END ===\n")

    return evaluation, feedback

def loop(
    task: str,
    evaluator_prompt: str,
    generator_prompt: str,
    dct_params: dict,
    n_max_iter = 5,
    debug_mode = True
    ) -> tuple[str, list[dict]]:
    """Keep generating and evaluating until requirements are met."""
    memory = []
    chain_of_thought = []

    thoughts, result = generate(generator_prompt, task, dct_params=dct_params)
    memory.append(result)
    chain_of_thought.append({"thoughts": thoughts, "result": result})

    i = 0
    while True:
        # =======================
        if debug_mode:
          time.sleep(20)
        # =======================

        evaluation, feedback = evaluate(evaluator_prompt, result, task, dct_params=dct_params)
        if evaluation == "PASS" or i <= n_max_iter:
            return result, chain_of_thought

        context = "\n".join([
            "Previous attempts:",
            *[f"- {m}" for m in memory],
            f"\nFeedback: {feedback}"
        ])

        # =======================
        if debug_mode:
          time.sleep(20)
        # =======================

        thoughts, result = generate(generator_prompt, task, context, dct_params=dct_params)
        memory.append(result)
        chain_of_thought.append({"thoughts": thoughts, "result": result})

        i += 1