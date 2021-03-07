import pickle
from assignment2 import *

with open("sida.pickle", "rb") as f:
    pi1, pi2, pi1_, pi2_, player1, player2, v1, v2, v1_vi, v2_vi = pickle.load(f)

# check values:

test_case_states = [
    (0, 0, 2, -2, 0), # this is from Reflection Question 2.
    (0, 1, 1, -1, -1), # this is the next state of Reflection Q2.
    (-2,0,0,0,2),
    (-2,0,0,2,0),
    (-2,0,2,0,0),
    (-2,2,0,0,0),
    (0,-1,-1,0,2),
    (-1,-1,0,1,1),
    (0,-1,1,-1,1),
    (0,2,-1,-1,0),
]
game = MiniGammon5Env()

def check_policy(states, policy):
    for state in states:
        game.set_state(state, [1], policy.token)
        a = policy.get_action(game)
        policy
        print(a)
def check_value(states, values):
    for state in states:
        v = values[state]
        print(v)

print("check pi1", "="*10)
check_policy(test_case_states, pi1)
print("check pi2", "="*10)
check_policy(test_case_states, pi2)
print("check pi1_", "="*10)
check_policy(test_case_states, pi1_)
print("check pi2_", "="*10)
check_policy(test_case_states, pi2_)
print("check player1", "="*10)
check_policy(test_case_states, player1)
print("check player2", "="*10)
check_policy(test_case_states, player2)

print("check v1", "="*10)
check_value(test_case_states, v1)
print("check v2", "="*10)
check_value(test_case_states, v2)

print("check v1 from value iteration", "="*10)
check_value(test_case_states, v1_vi)
print("check v2 from value iteration", "="*10)
check_value(test_case_states, v2_vi)
