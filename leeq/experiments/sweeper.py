import copy
import uuid
import itertools
from typing import Union, Iterable, Callable, Dict, Optional, List

from leeq.core import LeeQObject
from functools import partial

from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep


class SweepParametersSideEffectFunction(LeeQObject):
    """
    This class is for defining a sweep parameter side effect, by calling functions.

    The side effect means that the parameter being swept will affect the program by making a function call.
    """

    def __init__(self, function, argument_name, **kwargs):
        """
        Create a SweepParametersSideEffectFunction object from a function. The function will be called with the
        arguments and keyword arguments provided in the constructor, and the swept parameter.

        Parameters:
            function (function): The function to be called.
            argument_name (str): The name of the argument that will be set to the swept parameter.
            kwargs (dict): The keyword arguments to be passed to the function.
        """

        super().__init__(name='SideEffectFunction: ' + function.__name__)
        self._argument_name = argument_name
        self._function = partial(function, **kwargs)

    def __call__(self, parameter):
        """
        Call the function with the arguments and keyword arguments provided in the constructor, and the swept parameter.

        Parameters:
            parameter (Any): The swept parameter.
        """
        return self._function(**{self._argument_name: parameter})


class SweepParametersSideEffectAttribute(LeeQObject):
    """
    This class is for defining a sweep parameter side effect, by setting an attribute.

    The side effect means that the parameter being swept will affect the program by setting an attribute of an object.
    """

    def __init__(self, object_instance, attribute_name):
        """
        Create a SweepParametersSideEffectFunction object from a function. The function will be called with the
        arguments and keyword arguments provided in the constructor.

        Parameters:
            object_instance (object): The object whose attribute will be set.
            attribute_name (str): The name of the attribute that will be set to the swept parameter.
        """

        super().__init__(name='SideEffectFunction: ' + object_instance.__class__.__qualname__ + '.' + attribute_name)
        self._object_instance = object_instance
        self._attribute_name = attribute_name

    def __call__(self, parameter):
        """
        Update the attribute value with the parameter.

        Parameters:
            parameter (Any): The swept parameter.
        """
        setattr(self._object_instance, self._attribute_name, parameter)


class SweepParametersSideEffectFactory(LeeQObject):

    @classmethod
    def from_function(cls, function, argument_name, **kwargs):
        """
        Create a SweepParametersSideEffectFunction object from a function. The function will be called with the
        arguments and keyword arguments provided in the constructor, and the swept parameter.

        Parameters:
            function (function): The function to be called.
            argument_name (str): The name of the argument that will be set to the swept parameter.
            kwargs (dict): The keyword arguments to be passed to the function.
        """

        return SweepParametersSideEffectFunction(function, argument_name, **kwargs)

    @classmethod
    def func(cls, function, kwargs, argument_name):
        """
        Same as `from_function`, the arguments no longer accept arbitrary arguments, but has to pass in a dictionary.
        For compatibility reasons.
        """
        return cls.from_function(function=function, argument_name=argument_name, **kwargs)

    @classmethod
    def from_attribute(cls, method, *args, **kwargs):
        """
        Create a SweepParametersSideEffectFunction object from an attribute. The attribute will be set to the
        swept parameter.
        """
        return SweepParametersSideEffectAttribute(method, *args, **kwargs)

    @classmethod
    def attr(cls, method, *args, **kwargs):
        """
        Same as `from_attribute`, for compatibility reasons.
        """
        return cls.from_attribute(method, *args, **kwargs)


class Sweeper(LeeQObject):
    """
        This class is for defining a sweep parameter with side effects.

        The side effect means that the parameter being swept will affect the program by making a function call,
        or setting attribute of an object. For example, you can sweep a frequency of a qubit, and the side effect
        is that the qubit is set to that frequency. Before moving to the next frequency, the qubit is set to the
        frequency of the current step.

        Parameters:
    """

    def __init__(self, sweep_parameters: Union[Iterable, Callable],
                 params: List[Union[SweepParametersSideEffectAttribute, SweepParametersSideEffectFunction]],
                 n_kwargs: Optional[Dict] = None, child: 'Sweeper' = None, name: str = None):
        """
        Create a Sweeper object.

        Parameters:
            sweep_parameters (Union[Iterable, Callable]): The parameter to be swept. If it is an iterable, the parameter
             will be swept over the iterable. If it is a callable, the parameter will be swept over the return values
                of the callable. The callable must accept the arguments provided in the params argument.
            params (List[Union[SweepParametersSideEffectAttribute,SweepParametersSideEffectFunction]]): The side
                effects of the parameter being swept. The side effects are defined by the objects of the class
                SweepParametersSideEffectAttribute and SweepParametersSideEffectFunction.
            n_kwargs (Optional[Dict]): Optional. The keyword arguments to be passed to the callable n.
            child (Sweeper): Optional. The next sweeper in the chain.
            name (str): Optional. The name of the sweeper.
        """
        if name is None:
            name = f'Sweeper: {uuid.uuid4()}'
        super().__init__(name)

        if n_kwargs is None:
            n_kwargs = {}

        self._child = child  # This is the next sweeper in the chain
        if callable(sweep_parameters):
            sweep_parameters = sweep_parameters(**n_kwargs)
        self._sweep_parameters = sweep_parameters
        self._params = params
        # self._step = 0  # The current step of the sweeper
        # self._child_initialized = True

    @property
    def child(self):
        """
        The child sweeper in the chain.

        Returns:
            Sweeper: The child sweeper in the chain.
        """
        return self._child

    @property
    def shape(self):
        if self._child is None:
            return (len(self._sweep_parameters),)
        else:
            return (len(self._sweep_parameters),) + self._child.shape

    @property
    def sweep_parameters(self):
        """
        The parameters to be swept.

        Returns:
            Union[Iterable]: The parameter to be swept.
        """
        return self._sweep_parameters

    def add_to_chain(self, child):
        """
        Add a child sweeper to the chain.

        Parameters:
            child (Sweeper): The child sweeper to be added.
        """
        if self._child is None:
            self._child = child
        else:
            self._child.add_to_chain(child)

    def __add__(self, other):
        """
        Add a child sweeper to the chain.

        Parameters:
            other (Sweeper): The child sweeper to be added.
        """
        obj = copy.copy(self)
        obj.add_to_chain(other)
        return obj

    def __copy__(self):
        """
        Create a copy of the sweeper.

        Returns:
            Sweeper: The copy of the sweeper.
        """
        obj = self.__new__(Sweeper)
        obj._step = self._step
        obj._params = self._params
        obj._child = copy.copy(self._child)
        obj._sweep_parameters = copy.copy(self._sweep_parameters)
        obj._child_initialized = True
        return obj

    # def __iter__(self):
    #    self.reset()
    #    self._child_initialized = False
    #    return self

    # def __next__(self):
    #    child_finished = False
    #    if self._child is not None:
    #        if not self._child_initialized:
    #            child_value = self._child.reset()
    #            self._child_initialized = True
    #        else:
    #            try:
    #                child_value = next(self._child)
    #            except StopIteration:
    #                child_value = self._child.reset()
    #                self._child_initialized = True
    #                child_finished = True
    #    else:
    #        child_value = ()
    #        child_finished = True

    #    if child_finished:
    #        self._step += 1
    #        if self._step >= len(self._sweep_parameters):
    #            raise StopIteration

    #    self._execute_side_effects(self._sweep_parameters[self._step])
    #    return (self._step,) + child_value

    # def reset(self):
    #    self._step = 0
    #    self._execute_side_effects(self._sweep_parameters[self._step])
    #    result = (self._step,)
    #    if self._child is not None:
    #        result = result + self._child.reset()

    #    return result

    def _execute_side_effects(self, parameter):
        """
        Execute the side effects of the parameter being swept.

        Parameters:
            parameter (Any): The swept parameter.
        """
        for param in self._params:
            param(parameter)

    def execute_side_effects_by_step_no(self, step_no):
        """
        Execute the side effects of the parameter being swept.

        Parameters:
            step_no (int, list): The step no of the parameter.
        """

        if isinstance(step_no, int):
            self._execute_side_effects(self._sweep_parameters[step_no])

        assert len(step_no) == len(
            self.shape), f'The length of step_no {len(step_no)} must be equal to the length of shape {len(self.shape)}.'

        result = [self._execute_side_effects(self._sweep_parameters[step_no[0]])]

        if self._child is None:
            return result
        return result + self._child.execute_side_effects_by_step_no(step_no[1:])

    @classmethod
    def from_sweep_lpb(cls, slpb: LogicalPrimitiveBlockSweep):
        """
        Create a sweeper from a logical primitive block sweep.

        Parameters:
            slpb (LogicalPrimitiveBlockSweep): The logical primitive block sweep.

        Returns:
            Sweeper: The sweeper.
        """
        return cls(range(len(slpb)), params=[SweepParametersSideEffectFactory.func(
            slpb.set_selected, {}, 'selected')])
