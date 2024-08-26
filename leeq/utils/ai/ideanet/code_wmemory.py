import re

from leeq.utils.ai.ideanet.w_memory import WMemoryItem

"""
# Working memory items for code
"""


class CodeWMemoryItem(WMemoryItem):
    def __init__(self, code: str, tag="code"):
        super().__init__(code, tag=tag)

    def get_stimuli(self):
        do_stimuli = extract_func_parameters("do", self.content)
        comment_stimuli = extract_code_comment(self.content)
        stimuli = do_stimuli + comment_stimuli
        return stimuli


def init_code_for_wm(raw_instruction: str) -> str:
    split_instruction = raw_instruction.split("\n")
    code_items = []
    for inst in split_instruction:
        inst = inst.strip()
        if len(inst) == 0:
            code_items.append("")
            continue
        if inst.startswith("#"):
            code_items.append(inst)
            continue
        if "\n" in inst:
            code_items.append(f'do("""{inst}""")')
        elif "\"" in inst:
            inst = inst.replace("\'", "\\\'")
            code_items.append(f"do('{inst}')")
        else:
            code_items.append(f'do("{inst}")')
    code = "\n".join(code_items).strip()
    return code


def init_code_wm_item(raw_instruction: str) -> CodeWMemoryItem:
    code = init_code_for_wm(raw_instruction)
    return CodeWMemoryItem(code)


def extract_func_parameters(func_name, code):
    pattern = re.compile(rf'{func_name}\(\"((.|\n)*?)\"\s*\)', re.DOTALL)
    matches = pattern.findall(code)
    pattern2 = re.compile(rf'{func_name}\(\'((.|\n)*?)\'\s*\)', re.DOTALL)
    matches.extend(pattern2.findall(code))
    res = [x[0] for x in matches]
    res = [x.strip() for x in res]
    res = [x.strip("\"") for x in res]
    return res


def extract_code_comment(code) -> list[str]:
    """
    Extract all the comment in the form of # comment
    """
    pattern = re.compile(r"#\s*(.*)")
    comments = pattern.findall(code)
    docstring_pattern = re.compile(r'\"\"\"((.|\n)*?)\"\"\"', re.DOTALL)
    docstring_pattern2 = re.compile(r"\'\'\'((.|\n)*?)\'\'\'", re.DOTALL)
    docstrings = docstring_pattern.findall(code)
    docstrings.extend(docstring_pattern2.findall(code))
    docstrings = [x[0] for x in docstrings]
    return comments + docstrings


class CodeEditingItem(WMemoryItem):
    def __init__(self, original, new, note: str = ""):
        suggestion = f"""
Original: {original}
New: {new}
"""
        if note:
            suggestion += f"Note: {note}"
        self.original = original
        self.new = new
        self.note = note
        super().__init__(suggestion, "code_suggestion")

    def get_stimuli(self):
        res = [self.original, self.new]
        if len(self.note) > 0:
            res.append(self.note)
        return res
