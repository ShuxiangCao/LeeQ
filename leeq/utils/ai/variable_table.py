from __future__ import annotations
from types import ModuleType
from typing import Tuple, Optional

import numpy
from asteval import Interpreter


def empty_interpreter() -> Interpreter:
    """Create a new asteval interpreter instance with restricted numpy use and extended import capabilities.

    Returns:
        Interpreter: A new asteval.Interpreter instance configured without numpy and with import capabilities.
    """
    return Interpreter(use_numpy=False, config={'import': True, 'importfrom': True})


class VariableTable:
    """A table for managing variables within a dynamic execution context, supporting nested scopes."""

    names_in_empty_interpreter: Optional[set] = None

    def __init__(self):
        """Initialize an empty VariableTable."""
        self.variable_objs = {}
        self.variable_docs = {}
        self._parent_tables = []

    def add_variable(self, name: str, obj: object, docs: Optional[str] = None):
        """Add or update a variable along with optional documentation.

        Args:
            name (str): The name of the variable.
            obj (object): The object to store as a variable.
            docs (Optional[str]): Optional documentation string for the variable.
        """
        self.variable_objs[name] = obj
        self.variable_docs[name] = docs

    def get_variable(self, name: str) -> Tuple[object, str]:
        """Retrieve a variable and its documentation by name.

        Args:
            name (str): The name of the variable to retrieve.

        Returns:
            Tuple[object, str]: A tuple containing the variable and its documentation.
        """
        return self.variable_objs[name], self.variable_docs.get(name, "")

    def get_prompt(self, no_doc: bool = False) -> str:
        """Generate a formatted string representation of all variables in the current and parent tables.

        Args:
            no_doc (bool): Whether to exclude documentation from the prompt.

        Returns:
            str: A string representing all variables and optionally their documentation.
        """
        prompt_dict = {}
        for table in self._parent_tables:
            prompt_dict.update(table.get_local_prompt_dict(no_doc=no_doc))
        prompt_dict.update(self.get_local_prompt_dict(no_doc=no_doc))
        prompt_list = [f"{prompt}" for prompt in prompt_dict.values()]
        return "\n".join(prompt_list)

    def get_local_prompt(self) -> str:
        """Generate a formatted string of all local variables.

        Returns:
            str: A formatted string of all local variables with their representations.
        """
        prompt_dict = self.get_local_prompt_dict()
        prompt_list = [f"{prompt}" for prompt in prompt_dict.values()]
        return "\n".join(prompt_list)

    def get_local_prompt_dict(self, no_doc: bool = False) -> dict:
        """Generate a dictionary of variable names to formatted strings, optionally including documentation.

        Args:
            no_doc (bool): Whether to exclude documentation from the formatted output.

        Returns:
            dict: A dictionary mapping variable names to formatted strings.
        """
        prompt_dict = {}
        for name, obj in self.variable_objs.items():
            value = self.variable_objs[name]
            if no_doc:
                prompt_dict[name] = f"VarName:`{name}` {get_repr_in_prompt(value)}"
            else:
                doc = self.variable_docs.get(name, None)
                if doc is not None:
                    prompt_dict[name] = f"VarName:`{name}` Doc: {doc} {get_repr_in_prompt(value)}"
                else:
                    prompt_dict[name] = f"VarName:`{name}` {get_repr_in_prompt(value)}"
        return prompt_dict

    def is_empty(self) -> bool:
        """Check if the current table and all parent tables have any variables defined.

        Returns:
            bool: True if no variables are defined, False otherwise.
        """
        return len(self.get_prompt().strip()) == 0

    def is_local_empty(self) -> bool:
        """Check if the current table alone has any variables defined.

        Returns:
            bool: True if no local variables are defined, False otherwise.
        """
        return len(self.get_local_prompt().strip()) == 0

    def get_interpreter(self) -> Interpreter:
        """Get a configured interpreter instance with all variables from this table and parent tables.

        Returns:
            Interpreter: A configured asteval interpreter instance.
        """
        interpreter = empty_interpreter()
        self.add_to_interpreter(interpreter)
        return interpreter

    def interpret(self, code: str, assign_back: bool = True):
        """Execute code using an asteval interpreter and optionally assign back any new or modified variables.

        Args:
            code (str): The code to be executed.
            assign_back (bool): Whether to update the table with changes made during interpretation.
        """
        interpreter = self.get_interpreter()
        interpreter(code)
        if assign_back:
            self.assign_back(interpreter)

    def assign_back(self, interpreter: Interpreter):
        """Update the variable table with new or modified variables from the interpreter.

        Args:
            interpreter (Interpreter): The interpreter with potentially updated variables.
        """
        if VariableTable.names_in_empty_interpreter is None:
            VariableTable.names_in_empty_interpreter = set(empty_interpreter().symtable.keys())
        for name in interpreter.symtable:
            if name in VariableTable.names_in_empty_interpreter:
                continue
            table = self.find_table_for_symbol(name)
            if table is not None:
                table.variable_objs[name] = interpreter.symtable[name]
            else:
                self.variable_objs[name] = interpreter.symtable[name]

    def find_table_for_symbol(self, symbol: str) -> Optional[VariableTable]:
        """Search for the variable table that contains the specified symbol.

        Args:
            symbol (str): The symbol name to search for.

        Returns:
            Optional[VariableTable]: The table that contains the symbol, or None if not found.
        """
        if symbol in self.variable_objs:
            return self
        for parent_table in self._parent_tables:
            table = parent_table.find_table_for_symbol(symbol)
            if table is not None:
                return table
        return None

    def add_to_interpreter(self, interpreter: Interpreter) -> Interpreter:
        """Add all variables from this table and its parent tables to an interpreter.

        Args:
            interpreter (Interpreter): The interpreter to add variables to.

        Returns:
            Interpreter: The updated interpreter instance.
        """
        for parent_table in self._parent_tables:
            parent_table.add_to_interpreter(interpreter)
        for name, obj in self.variable_objs.items():
            interpreter.symtable[name] = obj
        return interpreter

    def new_child_table(self) -> VariableTable:
        """Create a new child variable table inheriting from this table.

        Returns:
            VariableTable: A new child table linked to the current table as a parent.
        """
        new_table = VariableTable()
        new_table._parent_tables = [self]
        return new_table

    def update_by_other_table(self, other_table: VariableTable):
        """Update the current table's variables and documentation by merging another table's content.

        Args:
            other_table (VariableTable): Another variable table whose contents are to be merged into this one.
        """
        self.variable_objs.update(other_table.variable_objs)
        self.variable_docs.update(other_table.variable_docs)

    def add_parent_table(self, parent_table: VariableTable):
        """Add a parent variable table to enable variable resolution in nested contexts.

        Args:
            parent_table (VariableTable): A parent variable table to be linked.
        """
        self._parent_tables.append(parent_table)


def get_repr_in_prompt(value):
    long_limit = 3
    if isinstance(value, int) or isinstance(value, float):
        return f"Value: {value}"
    elif isinstance(value, str):
        return f"Value: '{value}'"
    elif isinstance(value, tuple) or isinstance(value, list):
        res = "[" if isinstance(value, list) else "("
        if len(value) > long_limit:
            res += get_repr_in_prompt(value[0]) + ", " + get_repr_in_prompt(
                value[1]) + ", ..."
            res += ", " + get_repr_in_prompt(value[-1])
        else:
            res += ", ".join([get_repr_in_prompt(v) for v in value])
        res += "]" if isinstance(value, list) else ")"
        return f"Value: {res}"
    elif isinstance(value, dict):
        res = "{"
        dict_kv_list = [f"{k}: {get_repr_in_prompt(v)}" for k, v in value.items()]
        if len(dict_kv_list) > long_limit:
            res += ", ".join(dict_kv_list[:long_limit])
            res += ", ..."
            res += ", ".join(dict_kv_list[-1])
        else:
            res += ", ".join(dict_kv_list)
        res += "}"
        return f"Value: {res}"
    elif isinstance(value, numpy.ndarray):
        shape = value.shape
        return f"Type: numpy array, Shape: {shape}"
    elif isinstance(value, ModuleType):
        name = value.__name__.split(".")[-1]
        return f"Module: {name}"
    else:
        repr_str, classname = get_truncated_repr(value)
        return f"Value: {repr_str} Type: {classname}"


def get_truncated_repr(obj, limit=30):
    classname = obj.__class__.__name__
    repr_str = repr(obj)
    if len(repr_str) > limit:
        repr_str = repr_str[:limit] + "..." + repr_str[-1:]
    return repr_str, classname


def get_truncated_repr(obj: object, limit: int = 30) -> Tuple[str, str]:
    """Generate a truncated string representation of an object along with its class name.

    The string representation is truncated to a specified character limit to ensure it remains concise,
    especially useful for displaying complex objects in limited UI space. The truncation preserves the
    start and end of the string, providing a hint of the complete value.

    Args:
        obj (object): The object whose representation is to be truncated.
        limit (int): The maximum length of the truncated string representation. Defaults to 30 characters.

    Returns:
        Tuple[str, str]: A tuple containing the truncated string representation and the class name of the object.
    """
    classname = obj.__class__.__name__
    repr_str = repr(obj)
    if len(repr_str) > limit:
        repr_str = repr_str[:limit] + "..." + repr_str[-1:]
    return repr_str, classname
