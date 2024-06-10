import os.path

import markdown
import mllm
from bs4 import BeautifulSoup
from markdownify import markdownify
from mllm.utils import parallel_map

from leeq.experiments.ai.automation import FullyAutomatedExperiment
from leeq.utils.ai.code_indexer import imagine_applications, add_leeq_exp_to_ltm, \
    build_leeq_code_ltm


def extract_class_name(title, description):
    prompt = f"""
Based on the description of the experiment class `{title}`, please provide the name of the class. 
<description>
{description}
</description>
<instruction>
You should output a JSON dict with key "class_name" as a string.
</instruction>
"""
    chat = mllm.Chat(prompt)
    res = chat.complete(parse="dict", cache=True)
    return res["class_name"]


def extract_procedure_contents(markdown_path):
    with open(markdown_path, "r") as f:
        src = f.read()
    # Get html of the markdown
    html = markdown.markdown(src)
    # Parse the html
    soup = BeautifulSoup(html, "html.parser")
    procedures = []
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
        # Convert the html to markdown with sections start with #
        sibling_md = markdownify(sibling_html, heading_style="ATX")
        procedures.append((title, sibling_md))
    return procedures


def automated_experiment_factory(title_prompt):
    title, prompt = title_prompt
    name = extract_class_name(title, prompt)

    class AutomatedExperiment(FullyAutomatedExperiment):
        needing_situations = imagine_applications(title, prompt)

        def run(self, **kwargs):
            super().run(prompt, **kwargs)

    AutomatedExperiment.__name__ = name

    return AutomatedExperiment


def extract_procedures(markdown_path):
    procedure_contents = extract_procedure_contents(markdown_path)
    exp_classes = []
    for i, exp in parallel_map(automated_experiment_factory, procedure_contents,
                               title="Extracting procedures"):
        exp_classes.append(exp)
    return exp_classes


def extract_procedures_to_lt_memory(markdown_path, var_table, lt_memory):
    exp_classes = extract_procedures(markdown_path)
    for exp_class in exp_classes:
        add_leeq_exp_to_ltm(lt_memory, var_table, exp_class)


if __name__ == '__main__':
    lt_memory, var_table = build_leeq_code_ltm()
    from leeq.experiments.builtin.basic import calibrations

    root = os.path.dirname(calibrations.__file__)
    extract_procedures_to_lt_memory(root + "/procedures/calibration.md", var_table,
                                    lt_memory)
