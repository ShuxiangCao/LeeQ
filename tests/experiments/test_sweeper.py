import pytest
from leeq.experiments.sweeper import SweepParametersSideEffectFunction, SweepParametersSideEffectAttribute, Sweeper, \
    SweepParametersSideEffectFactory

from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.core.primitives.built_in.simple_drive import SimpleDriveCollection


# Dummy function for testing
def dummy_function(arg1, arg2=0):
    return arg1 + arg2


# Test case for __init__ and __call__
def test_SweepParametersSideEffectFunction():
    sweep_func = SweepParametersSideEffectFunction(
        dummy_function, 'arg1', arg2=5)
    assert sweep_func.__call__(10) == {'arg1': 10}

    # Test with different values
    assert sweep_func.__call__(20) == {'arg1': 20}

    # Test with overridden kwargs
    sweep_func = SweepParametersSideEffectFunction(
        dummy_function, 'arg1', arg2=5)
    assert sweep_func.__call__(30) == {'arg1': 30}


class DummyObject:
    pass


def test_SweepParametersSideEffectAttribute():
    dummy_obj = DummyObject()
    sweep_attr = SweepParametersSideEffectAttribute(dummy_obj, 'attribute')
    sweep_attr.__call__(10)
    assert dummy_obj.attribute == 10

    # Test with different values
    sweep_attr.__call__(20)
    assert dummy_obj.attribute == 20


def test_SweepParametersSideEffectFactory():
    dummy_obj = DummyObject()
    sweep_func = SweepParametersSideEffectFactory.from_function(
        dummy_function, 'arg1', arg2=5)
    assert sweep_func.__call__(10) == {'arg1': 10}

    sweep_func = SweepParametersSideEffectFactory.func(
        dummy_function, {'arg2': 5}, 'arg1')
    assert sweep_func.__call__(10) == {'arg1': 10}

    sweep_attr = SweepParametersSideEffectFactory.from_attribute(
        dummy_obj, 'attribute')
    assert sweep_attr.__call__(10) == {'attribute': 10}
    assert dummy_obj.attribute == 10

    sweep_attr = SweepParametersSideEffectFactory.attr(dummy_obj, 'attribute')
    assert sweep_attr.__call__(20) == {'attribute': 20}
    assert dummy_obj.attribute == 20


def dummy_sweep_func():
    return range(5)


def test_Sweeper():
    dummy_obj = DummyObject()
    params = [
        SweepParametersSideEffectFunction(dummy_function, 'arg1', arg2=5),
        SweepParametersSideEffectAttribute(dummy_obj, 'attribute')
    ]
    sweeper = Sweeper(dummy_sweep_func, params)

    assert sweeper.shape == (5,)
    assert sweeper.sweep_parameters == range(5)

    sweeper_copy = sweeper.__copy__()
    assert sweeper_copy.shape == (5,)
    assert sweeper_copy.sweep_parameters == range(5)

    # sweeper_iter = sweeper.__iter__()
    # sweeper.reset()
    # assert sweeper._step == 0

    sweeper2 = Sweeper(dummy_sweep_func, params)
    chained_sweeper = sweeper + sweeper2
    assert chained_sweeper.shape == (5, 5)

    sweeper.add_to_chain(sweeper2)
    assert sweeper._child == sweeper2

# def test_Sweeper_integration():
#    dummy_obj_1 = DummyObject()
#    dummy_obj_2 = DummyObject()
#
#    side_effect_values = []
#
#    swp = Sweeper(range(3), [SweepParametersSideEffectFactory.attr(dummy_obj_1, 'attr')])
#    swp2 = Sweeper(range(5), [SweepParametersSideEffectFactory.attr(dummy_obj_2, 'attr')])
#
#    swp3 = swp + swp2
#
#    for result in swp3:
#        side_effect_values.append((dummy_obj_1.attr, dummy_obj_2.attr))
#
#    result = [x for x in swp3]
#
#    assert len(side_effect_values) == 15
#    assert len(result) == 15
#    assert result == side_effect_values
#
#    assert result == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
#                      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
#                      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
#
#    swp3 = Sweeper(range(2), [])
#    swp4 = swp + swp2 + swp3
#
#    result = [x for x in swp4]
#    assert len(result) == 30

# def test_sweep_lpb_to_sweeper():
#    c1 = SimpleDriveCollection(
#        name='f12', parameters={
#            'type': 'SimpleDriveCollection',
#            'freq': 4144.417053428905,
#            'channel': 2,
#            'shape': 'BlackmanDRAG',
#            'amp': 0.21323904814245054 / 5 * 4,
#            'phase': 0.,
#            'width': 0.025,
#            'alpha': 425.1365229849309,
#            'trunc': 1.2
#        }
#    )
#
#    lpb_1 = c1['X']
#    lpb_2 = c1['Y']
#    lpb_3 = c1['Yp']
#
#    lpb_list = [lpb_1, lpb_2, lpb_3]
#
#    lpb = LogicalPrimitiveBlockSweep(lpb_list)
#    swp = Sweeper.from_sweep_lpb(lpb)
#
#    assert swp.shape == (3,)
#
#    for i, result in enumerate(swp):
#        a = repr(lpb_list[i])
#        b = lpb.current_lpb
#        assert a == b
#
