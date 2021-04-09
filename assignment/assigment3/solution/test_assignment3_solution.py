import assignment3_solution as a3
import pytest

def test_one():
    assert 1==1

def test_get_action_probility_for_state_1():
    """ Testing get_action_probility_for_state(), Case 1 
    Reset the environment and calculate the probability of a random action for a RandomAgent
    """
    game = a3.MiniGammon5Env()
    game.reset()
    state = game.get_state()

    agent = a3.RandomAgent()
    action = agent.get_action(game)

    # TODO: ok, we've detected some problem, we have a TypeError. it should be fixed later.
    # maybe we should check and correct the format of the state in `get_action_probility_for_state()`.
    ret = a3.get_action_probility_for_state(state, action, agent)
    print(ret)

if __name__=='__main__':
    # the CLI command `pytest` can start all tests, but sometimes we want to debug step-by-step, so we can put it here.
    test_get_action_probility_for_state_1()