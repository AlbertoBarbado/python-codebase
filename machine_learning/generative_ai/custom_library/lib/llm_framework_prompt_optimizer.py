import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable
from lib.utils import extract_xml
from lib.llm_tools import llm_call

def execute_task(
    prompt: str,
    tasks: List[str],
    context: str,
    dct_params: dict,
    debug_mode=True
    ) -> tuple[List[str], List[str]]:
    """Execute a batch of tasks using the given prompt and context."""

    # Fixed prompt
    output_structure = """
    Output your answer concisely in the following XML format, using only these elements, and without repeating input information:

    <thoughts> Your understanding of the task and how do you plan to solve it </thoughts>
    <response> Your answer here </response>
    """

    results = []
    thoughts_list = []

    for task in tasks:
        full_prompt = f"{prompt}\n{output_structure}\n{context}\nTask: {task}" if context else f"{prompt}\n{output_structure}\nTask: {task}"
        print("\n=== TASK EXECUTION INPUT START ===")
        print(f"Full prompt:\n{full_prompt}\n")
        print("\n=== TASK EXECUTION INPUT END ===")

        response = llm_call(full_prompt, **dct_params)
        thoughts = extract_xml(response, "thoughts")
        result = extract_xml(response, "response")

        print("\n=== TASK EXECUTION OUTPUT START ===")
        print(f"Raw output:\n{response}")
        print(f"Thoughts:\n{thoughts}\n")
        print(f"Generated:\n{result}")
        print("=== TASK EXECUTION OUTPUT END ===\n")

        thoughts_list.append(thoughts)
        results.append(result)

        # Optional debug delay
        if debug_mode:
            time.sleep(20)

    return thoughts_list, results

def refine_prompt(
    input_system_prompt: str,
    tasks: List[str],
    outputs: List[str],
    memory: List[str],
    targets: List[str],
    dct_params: dict
    ) -> str:
    """Refine the system prompt based on the system prompt, tasks, outputs by another LLM, memory and target outputs"""

    # Fixed prompt
    engineer_prompt = """
      You are a prompt engineering expert.

      1. Task:
      * Given a <input_system_prompt> for another LLM, the <tasks> that the LLM is trying to solve,
      the <generated_outputs> for that tasks following that input system prompt,
      the <target_outputs> that should've been generated, and the <memory> of previous recommendations
      that you have provided, propose an improved <input_system_prompt>.

      2. Notes:
      * The new base prompt proposed should be generic enough for approaching that task even with different input data.
      * Thus, do not use specific information about the input data within the Tasks.
      * The new base prompt can include aspects such as synthetic examples for improving it, text refinement, task clarification...
      * You have memory information on previous attempts: improvements previously proposed and the output obtained.

      3. Output format:
      * Output your answer concisely in the following XML format, using only these elements:

      <evaluation> PASS, NEEDS_IMPROVEMENT, or FAIL </evaluation>
      <thoughts> Your understanding of the task and feedback and how you plan to improve </thoughts>
      <refined_prompt> Improved prompt </refined_prompt>
    """

    '''
    context = "\n".join([
        "Input system prompt:",
        *[input_system_prompt],
        "\nTasks:",
        *[f"- {task}" for task in tasks],
        "\nGenerated outputs:",
        *[f"- {output}" for output in outputs],
        "\nMemory:",
        *[f"- {m}" for m in memory],
        "\nTarget outputs:",
        *[f"- {target}" for target in targets]
    ])

    full_prompt = f"{engineer_prompt}\n{context}"
    '''
    context = f"""
    4. Input information (results from another LLM):
    <input_system_prompt> {input_system_prompt} </input_system_prompt>
    <tasks> {tasks} </tasks>
    <generated_output> {outputs} </generated_output>
    <memory> {memory} </memory>
    <target_outputs> {targets} </target_outputs>
    """

    full_prompt = f"{engineer_prompt}\n{context}"

    print("\n=== PROMPT ENGINEERING INPUT START ===")
    print(f"Full prompt:\n{full_prompt}\n")
    print("\n=== PROMPT ENGINEERING INPUT END ===")

    response = llm_call(full_prompt, **dct_params)
    refined_prompt = extract_xml(response, "refined_prompt")
    evaluation = extract_xml(response, "evaluation")

    print("\n=== PROMPT ENGINEERING OUTPUT START ===")
    print(f"Raw output:\n{response}")
    print(f"Refined Prompt:\n{refined_prompt}")
    print("=== PROMPT ENGINEERING OUTPUT END ===\n")

    return evaluation, refined_prompt

def iterative_task_execution(
    tasks: List[str],
    initial_prompt: str,
    target_outputs: List[str],
    dct_params: dict,
    n_max_iter: int = 5,
    debug_mode: bool = True
) -> str:
    """Iteratively execute and refine the batch of tasks until the targets are achieved."""
    memory = []
    chain_of_thought = []

    current_system_prompt = initial_prompt
    context = "" # We leave it empty by default

    for iteration in range(n_max_iter):
        print(f"\n=== ITERATION {iteration + 1} START ===")
        print("*"*50)
        print(f"current_system_prompt: {current_system_prompt}")
        print("*"*50)

        # Execute the tasks
        thoughts_list, results = execute_task(
            prompt = current_system_prompt,
            tasks = tasks,
            context = context,
            dct_params = dct_params
        )
        memory.extend(results)
        chain_of_thought.append({"iteration": iteration + 1, "thoughts": thoughts_list, "results": results})

        # Refine the prompt using the prompt engineer
        evaluation, output_prompt = refine_prompt(
            input_system_prompt = current_system_prompt,
            tasks = tasks,
            memory = memory,
            outputs = results,
            targets = target_outputs,
            dct_params = dct_params
        )

        if evaluation == 'PASS':
          return current_system_prompt
        else:
          current_system_prompt = output_prompt

        # Optional debug delay
        if debug_mode:
            time.sleep(20)

        '''
        # Update context for the next iteration
        context = "\n".join([
            "Previous results:",
            *[f"- {m}" for m in memory]
        ])
        '''

    print("\n=== MAX ITERATIONS REACHED ===")
    return current_system_prompt

def prepare_input(
    dct_input: Dict,
    input_key: str,
    target_key: str,
    max_inputs_per_prompt = 1 # Set the maximum number of input instances per prompt
    ):
    batch = [{input_key: sample[input_key], target_key: sample[target_key]} for sample in dct_input]

    # Prepare tasks and target summaries based on max_docs_per_prompt
    tasks = []
    target_texts = []

    for start_idx in range(0, len(batch), max_inputs_per_prompt):
        sub_batch = batch[start_idx:start_idx + max_inputs_per_prompt]

        task_text = "Apply the task for the following instances:"
        for i, item in enumerate(sub_batch):
            task_text += f"\n{i + 1}. {item[input_key]}"
        tasks.append(task_text)

        target_text = "The target references are:"
        for i, sample in enumerate(sub_batch):
            target_text += f"\n{i + 1}. {sample[target_key]}"
        target_texts.append(target_text)

    return tasks, target_texts