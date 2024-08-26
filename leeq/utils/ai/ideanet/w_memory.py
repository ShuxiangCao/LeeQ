from __future__ import annotations

from typing import List, TYPE_CHECKING, Set

import numpy as np
from mllm import get_embeddings

if TYPE_CHECKING:
    from .lt_memory import RecallResult


class WMemoryItem:
    """
    An item inside the working memory.
    """

    def __init__(self, content: str | List[WMemoryItem], tag: str | None = "context",
                 lifetime: int = -1, attrs: dict = None):
        self.content: str | List[WMemoryItem] = content
        self.tag: str = tag
        self.lifetime: int = lifetime
        self.no_stimuli: bool = False
        self.no_prompt: bool = False
        self.in_prompt_order: int = 0  # the smaller the in_prompt_order, the earlier it is shown in the prompt
        self.attrs: dict[str, str] = {}  # not sure if the type of items is always str

    def get_in_prompt_format(self) -> str | None:
        if self.no_prompt:
            return None

        attrs_in_prompt = []
        if len(self.attrs) > 0:
            for key, value in self.attrs.items():
                if key.startswith("_"):
                    continue
                attrs_in_prompt.append(f" {key}='{value}'")
        if len(attrs_in_prompt) > 0:
            attrs_in_prompt = "".join(attrs_in_prompt)
        else:
            attrs_in_prompt = ""

        if isinstance(self.content, str):
            content_in_prompt = self.content
        else:
            content_in_prompt = render_items(self.content)
        if self.tag is not None:
            return f"<{self.tag}{attrs_in_prompt}>\n{content_in_prompt}\n</{self.tag}>"
        else:
            if attrs_in_prompt != "":
                return f"<{attrs_in_prompt}>\n{content_in_prompt}</>"
            return content_in_prompt

    def set_no_stimuli(self):
        self.no_stimuli = True
        return self

    def set_no_prompt(self):
        self.no_prompt = True
        return self

    def set_order(self, order: int):
        """
        Set the order of appearance of this item inside the working memory.
        """
        self.in_prompt_order = order
        return self

    def get_stimuli(self) -> list[str]:
        if self.no_stimuli:
            return []
        if isinstance(self.content, str):
            split = self.content.split("\n")
            split = [item.strip() for item in split]
            split = [item for item in split if len(item) > 0]
            return split
        else:
            res = []
            for item in self.content:
                res.extend(item.get_stimuli())
            return res

    def __str__(self):
        in_prompt = self.get_in_prompt_format()
        if in_prompt is None:
            return "<hidden in prompt/>"
        return in_prompt

    def set_lifetime(self, lifetime) -> Self:
        if isinstance(lifetime, list):
            # sample from the list
            lifetime = np.random.choice(lifetime)
        self.lifetime = lifetime
        return self


class WMemorySuppressingItem(WMemoryItem):
    def __init__(self, idea, lifetime=3, tag=None):
        self.idea = idea
        super().__init__("", "", lifetime)
        self.tag = tag

    def get_in_prompt_format(self) -> str | None:
        return None

    def get_stimuli(self) -> list[str]:
        return []


class WMemoryHiddenItem(WMemoryItem):
    def __init__(self, stimuli: List[str], lifetime=-1):
        self.stimuli = stimuli
        super().__init__("", "", lifetime)

    def get_in_prompt_format(self) -> str | None:
        return None

    def get_stimuli(self) -> list[str]:
        return self.stimuli


class WMemoryNoStimuliItem(WMemoryItem):
    def __init__(self, content, tag="context", lifetime=-1):
        super().__init__(content, tag, lifetime)

    def get_stimuli(self) -> list[str]:
        return []


def render_items(items: List[WMemoryItem], tags_to_ignore=None) -> str:
    if tags_to_ignore is None:
        tags_to_ignore = []
    res = []
    items_included = []
    for item in items:
        if item.no_prompt or item.tag in tags_to_ignore:
            continue
        items_included.append(item)
    for item in sorted(items_included, key=lambda x: x.in_prompt_order):
        item_in_prompt = item.get_in_prompt_format()
        if item_in_prompt is not None:
            res.append(item_in_prompt)
    res = "\n".join(res)
    return res


class WorkingMemory:
    def __init__(self):
        self._items: List[WMemoryItem] = []

        self.cached_stimuli = None
        self.cached_stimuli_embeddings = None
        self.cached_in_prompt_str = None
        self.ticks: int = 0

    def get_in_prompt_format(self, tag=None, tags_to_ignore: list[str] = None) -> str:
        res = render_items(self._items, tags_to_ignore)
        if tag is None:
            return res
        return f"""<{tag}>
{res}
</{tag}>"""

    def _get_stimuli(self):
        res = []
        for item in self._items:
            if item.no_stimuli:
                continue
            res.extend(item.get_stimuli())
        return res

    @property
    def stimuli(self):
        if self.cached_stimuli is None:
            self.cached_stimuli = self._get_stimuli()
        return self.cached_stimuli

    @property
    def stimuli_embeddings(self):
        if self.cached_stimuli_embeddings is None:
            self.cached_stimuli_embeddings = np.array(get_embeddings(self.stimuli))
        return self.cached_stimuli_embeddings

    @property
    def in_prompt_str(self) -> str:
        if self.cached_in_prompt_str is None:
            self.cached_in_prompt_str = self.get_in_prompt_format()
        return self.cached_in_prompt_str

    def add_item(self, item: WMemoryItem) -> None:
        self._items.append(item)
        self.refresh_cache()

    def refresh_cache(self) -> None:
        self.cached_stimuli = None
        self.cached_stimuli_embeddings = None
        self.cached_in_prompt_str = None

    def add_content(self, content, tag="idea") -> WMemoryItem:
        item = WMemoryItem(content, tag)
        self.add_item(item)
        return item

    def tick(self) -> None:
        self.ticks += 1
        items_to_remove = []
        for item in self._items:
            if item.lifetime == 0:
                items_to_remove.append(item)
            if item.lifetime > 0:
                item.lifetime -= 1
        for item in items_to_remove:
            self._items.remove(item)
        self.refresh_cache()

    def get_suppressed_ideas(self) -> list[WMemorySuppressingItem]:
        suppressed_ideas = []
        for item in self._items:
            if isinstance(item, WMemorySuppressingItem):
                if item.lifetime > 0:
                    suppressed_ideas.append(item.idea)
        return suppressed_ideas

    def has_tag(self, tag) -> bool:
        for item in self._items:
            if item.tag == tag:
                return True
        return False

    def update_by_recall_res(self, recall_res: RecallResult, to_tick: bool = True) -> None:
        """Update working memory using the response from recalling a round of ideas."""
        recall_res.update_wm_from_res(self)
        if to_tick:
            self.tick()

    def extract_tag_contents(self, tag: str) -> List[str]:
        """Extract the contents of wm items tagged with <tag>. Returns empty list if nothing is found.

        :param tag: The tag to retrieve from.
        :return: list of extracted contents.
        """

        items = self.extract_tag_items(tag)
        return [item.content for item in items]

    def extract_tag_items(self, tag: str) -> List[WMemoryItem]:
        """Extract the items tagged with <tag>. Returns empty list if nothing is found.

        :param tag: The tag to retrieve from.
        :return: list of extracted items.
        """

        items = []
        for item in self._items:
            if item.tag == tag:
                items.append(item)
        return items

    def remove_item_by_tags(self, tags: List[str] | Set[str]):
        items_to_remove = []
        for item in self._items:
            if item.tag in tags:
                items_to_remove.append(item)
        for item in items_to_remove:
            self._items.remove(item)
        self.refresh_cache()
