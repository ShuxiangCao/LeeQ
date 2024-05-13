def visual_analyze_prompt(prompt: str):
    """
    Decorator function for the functions that used to visualize data of the class.
    It is used to register the prompt to analyze the data.

    Parameters:
        prompt (str): The prompt to be registered.

    Returns:
        Any: The return value of the function.
    """

    def inner_func(func):
        """
        Decorator function for the functions that used to visualize data of the class.
        It is used to register the prompt to analyze the data.
        The function must be a method of a LoggableObject.

        Parameters:
            func (function): The function to be registered.

        Returns:
            Any: The same function.
        """
        func._visual_prompt = prompt

        return func

    return inner_func

