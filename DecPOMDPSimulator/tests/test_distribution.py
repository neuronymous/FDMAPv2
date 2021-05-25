from DecPOMDPSimulator.Distribution import Distribution
import pytest


def test_sample():
    d = Distribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
    v = d.sample()
    assert v in ['a', 'b','c']

def test_sample_with_random_token():
    d = Distribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
    v = d.sample(random_token=0.8)
    assert v == 'a'
    v = d.sample(random_token=0.51)
    assert v == 'a'
    v = d.sample(random_token=0.49)
    assert v == 'b'
    v = d.sample(random_token=0.21)
    assert v == 'b'
    v = d.sample(random_token=0.19)
    assert v == 'c'
    v = d.sample(random_token=0.0)
    assert v == 'c'
