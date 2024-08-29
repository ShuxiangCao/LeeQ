from __future__ import annotations

import html
from typing import Any

from mllm.utils.logger import Logger
from mllm.display.show_html import show_json_table


class RecallLogger(Logger):
    active_loggers: list[Any] = []

    def __init__(self, disable=False):
        super().__init__(disable)

    def display_log(self):
        contents = [log for log in self.log_list]
        filenames = [caller_name.split("/")[-1] for caller_name in self.caller_list]
        info_list = []
        for i in range(len(contents)):
            content = contents[i]
            info_list.append({
                "filename": filenames[i],
                "content": content
            })
        show_json_table(info_list)


def to_log_item(item, title=None):
    res = []
    if title is not None:
        res.append(f"<h3>{title}</h3>")
    if isinstance(item, list):
        # make an ol list
        res.append('<ol>')
        for sub_item in item:
            res.append(f'<li>{to_log_item(sub_item)}</li>')
        res.append('</ol>')
    else:
        item_str = str(item)
        if item_str is not None:
            content = html.escape(str(item)).replace("\n", "<br/>")
            res.append(content)
    res.append("<br/>")
    return "".join(res)
