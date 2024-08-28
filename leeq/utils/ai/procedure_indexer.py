import inspect
import os.path
from pprint import pprint

import markdown
import mllm
from bs4 import BeautifulSoup
from markdownify import markdownify
from mllm import Chat
from mllm.utils import parallel_map
from mllm.utils.maps import p_map

from leeq.utils.ai.ideanet.lt_memory import IdeaResult, LongTermMemory, EmbedIdea
from leeq.experiments.ai.automation import FullyAutomatedExperiment
from leeq.utils.ai.code_indexer import imagine_applications, add_leeq_exp_to_ltm, \
    build_leeq_code_ltm
from leeq.utils.ai.ideanet.w_memory import WorkingMemory
from leeq.utils.ai.staging.stage_execution import get_codegen_wm
from leeq.utils.ai.variable_table import VariableTable


def extract_procedure_contents(markdown_path):
    with open(markdown_path, "r") as f:
        src = f.read()
    # Get html of the markdown
    html = markdown.markdown(src)
    # Parse the html
    soup = BeautifulSoup(html, "html.parser")
    title_html_list = []
    # Find the contents between <h1>
    for h1 in soup.find_all("h1"):
        siblings = []
        title = h1.text
        # Get the following siblings of h1
        for sibling in h1.next_siblings:
            # If the sibling is a tag, break the loop
            if sibling.name == "h1":
                break
            siblings.append(sibling)
        sibling_html = "".join([str(sibling) for sibling in siblings])
        title_html_list.append((title, sibling_html))
        # Convert the html to markdown with sections start with #
        #sibling_md = markdownify(sibling_html, heading_style="ATX")
        #procedures.append((title, sibling_md))
    procedure_list = []
    for title, sibling_html in title_html_list:
        # extract header with title Steps
        steps = ""
        background = ""
        for h2 in BeautifulSoup(sibling_html, "html.parser").find_all("h2"):
            if h2.text == "Steps":
                siblings = []
                # get all siblings until next h2
                for sibling in h2.next_siblings:
                    if sibling.name == "h2":
                        break
                    siblings.append(sibling)
                steps = "".join([str(sibling) for sibling in siblings])
                steps = markdownify(steps, heading_style="ATX").strip()
            elif h2.text == "Background":
                siblings = []
                for sibling in h2.next_siblings:
                    if sibling.name == "h2":
                        break
                    siblings.append(sibling)
                background = "".join([str(sibling) for sibling in siblings])
                background = markdownify(background, heading_style="ATX").strip()
        procedure_list.append({
            "title": title,
            "background": background,
            "steps": steps,
        })
    return procedure_list



class LeeQExpDocIdea(EmbedIdea):
    def __init__(self, title: str, steps: str, background: str, embed_src: list[str]):
        self.title = title
        self.steps = steps
        self.background = background
        super().__init__(f"LeeQExpDocIdea for {title}", embed_src)

    def run_idea(self, w_memory: WorkingMemory) -> IdeaResult:
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_items("available_variables")
        if len(available_variables) == 0:
            available_variables = ""
            var_table = VariableTable()
        else:
            available_variables = available_variables[0]
            var_table = available_variables.attrs["_table_obj"]
        prompt = f"""
You are trying to decomposing the following instruction for implementation.
<input_instruction>
{instruction}
</input_instruction>
You should generate instructions based on the following existing way of decomposing the task.
<example>
<instruction>
{self.title}
</instruction>
The above instruction can be decomposed into the following steps:
<steps>
{self.steps}
</steps>
The background for the above instruction is as follows:
<background>
{self.background}
</background>
</example>
<requirements>
You are required to output a JSON dict with the following keys
- "analysis" (string): An analysis of the relation between the input_instruction and the example, as well as how to decompose the input_instruction based on the example you have. You should notice that the input_instruction might be totally irrelevant to the example. You should point out whether the example provides a proper way to decompose the input_instruction, so that you don't need to modify the steps in the example a lot. You should also notice that the input_instruction might involve additional steps beyond the scope of the steps in example.
- "proper" (bool): Whether the example provides a proper way to decompose the input_instruction. If not proper, you can omit the next key.
- "broader" (bool): Whether the input_instruction might involve additional steps beyond the scope of the steps in example.
- "decomposed_steps" (string): The decomposed steps of input_instruction based on the <steps> in the example. You should only modify the existing <steps>. You should not add new steps. You should make minimal change to the existing <steps>.
- "certain_step" (bool): Whether the input_instruction matches a certain step in the example rather rather than many steps.
- "annotation" (string): A concise annotation to that indicate what will be implemented based on the decomposed steps.
</requirements>
"""
        chat = Chat(prompt)
        res = chat.complete(parse="dict", expensive=True)
        if not res["proper"] or res["broader"] or res["certain_step"]:
            return IdeaResult(self, False)
        #pprint(res)
        #print(res["decomposed_steps"])
        #print("Analysis")
        #print(res["analysis"])
        arg_in_code = []
        for arg in var_table.variable_objs:
            arg_in_code.append(f", {arg}={arg}")
        arg_in_code = "".join(arg_in_code)
        annotation_in_prompt = res["annotation"].replace("\n", "\n# ")
        code_suggestion = f'''
# {annotation_in_prompt}
# AutoRun function will execute instructions passed to it
AutoRun("""{res["decomposed_steps"]}""" {arg_in_code})        
'''
        #print(code_suggestion)
        idea_res = IdeaResult(self, True)
        idea_res.add_new_wm_content(code_suggestion, tag="code_suggestion")
        return idea_res

def imagine_applications_for_doc(title, background):
    prompt = f"""
You are trying to produce imperative sentences that would invoke the execution a certain experiment based on its title and background description.
<title>
{title}
</title>
<background>
{background}
</background>
<example>
Here are a few of examples of imperative sentences:
- Run the calibration experiment with duts=duts and start=0.0
- Calibrate `duts` 
- Please execute the MultiSingleQubitMultilevel experiment with end=2.0
- Do the AllXY drag experiment.
</example>
<instruction>
You should output a JSON dict. The keys should be string of indices of the sentences and the values should be the sentences. 
Each sentence should be complete and independent. Name of the experiment should be transformed to natural language and be mentioned.
The sentences should be imperative and should be based on the documentation.
You should output 4 sentences.
</instruction>"""
    chat = Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    values = list(res.values())
    return values

def generate_idea_from_procedure(procedure):
    title = procedure["title"]
    background = procedure["background"]
    steps = procedure["steps"]
    embed_src = imagine_applications_for_doc(title, background)
    idea = LeeQExpDocIdea(title, steps, background, embed_src + [title])
    return idea

def extract_procedures_to_lt_memory(markdown_path, lt_memory):
    procedures = extract_procedure_contents(markdown_path)
    for procedure, idea in p_map(generate_idea_from_procedure, procedures):
        lt_memory.add_idea(idea)


if __name__ == '__main__':
    from leeq import experiments as exp
    from leeq.utils.ai.ideanet.lt_memory import LongTermMemory
    from mllm.provider_switch import set_default_to_anthropic

    #set_default_to_anthropic()
    lt_memory = LongTermMemory()
    root = os.path.dirname(exp.__file__)
    extract_procedures_to_lt_memory(root + "/procedures/calibration.md", lt_memory)
    var_table = VariableTable()
    var_table.add_variable("dut", "DUT", "The device under test")
    #inst = "Do a complete Calibrating Single Qubit `dut`, in which set step=0.001 for the gate Amplitude Calibration. "
    inst = "Conduct a gate amplitude calibration on `dut_1`"
    wm = get_codegen_wm(inst, var_table=var_table)
    print(wm.get_in_prompt_format())
    lt_memory.recall_by_wm(wm)
