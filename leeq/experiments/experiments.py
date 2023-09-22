from labchronicle import LoggableObject


class Experiment(LoggableObject):
    """
    Base class for all experiments.

    An experiment contains the script to execute the experiment, analyze the data and visualize the result.

    1. Scripts execution
        To allow labchronicle to log the experiment, the main experiment script should be written in the `run` method.
        The `run` method should be decorated with the `labchronicle.log_and_record` method to log the events in the
         experiment. The decorator will save the arguments and return values of the `run` method and the entire object
          to the log.
        The `run` method will always be executed at the end of `__init__` to start the experiment.

    2. Data analysis
        The data analysis can be written in any arbitrary method, and suggested to run in a separate method than `run`,
         ideally the first few lines of the visualization code. This is because if the data analysis failed, it may
         crash the program before labchronicle can log the experiment data.

    3. Visualization
        The visualization code should live in a separate function and decorated by `labchronicle.browser_function`, so
         that the function will be executed when the experiment is finished execution in Jupyter notebook. It also
         allows the function to be executed later when data loaded from the log file.
    """

    def __init__(self):
        """
        Initialize the experiment.
        """
        super().__init__()

        # Run the experiment
        self.run()

        # Check if we need to plot
        # TODO: implement the check after the environment is set up.
        for func, args, kwargs in self._browse_functions:
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                self.logger.warning(f'Error when executing {func.__qualname__} with parameters ({args},{kwargs}): {e}')

    def run(self):
        """
        The main experiment script. Should be decorated by `labchronicle.log_and_record` to log the experiment.
        """
        raise NotImplementedError()
