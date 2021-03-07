import math
import numpy as np
np.random.seed(1234)

from abc import abstractmethod, ABC
from collections import Counter
from typing import Tuple, List, Dict, Any, Union

#%% [markdown]
# # Programming "Assignemnt" 2
# The goal of this second programming assignment is to give you experience computing value functions.
# For this assignment, you will be using a constrained version of backgammon, MiniGammon5

#%% [markdown]

# # Environments and helper functions

class MoveError(Exception): pass

def same_sign(a, b):
    return math.copysign(1, a) == math.copysign(1, b)

class BackgammonEnv(ABC):

    get_sign = {'x' : -1, 'o': 1}    


    def __init__(self, size=24, ntokens=15, homesize=6, ndice=2, die_nsides=6):
        self.turn = 'x'
        self.size = size
        self.ntokens = ntokens
        self.homesize = homesize
        self.ndice = ndice
        self.die_nsides = die_nsides

        self.dice = self.roll()
        self.board = self.get_start_board()
        self.out = None


    def set_state(self, board: Tuple[int], dice: List[int], turn=None) -> None:
        if turn: self.turn = turn
        self.board = board
        self.dice = dice
        pos = sum([t for t in board if t > 0])
        neg = abs(sum([t for t in board if t < 0]))
        self.out = {'x' : self.ntokens - neg, 'o': self.ntokens - pos}
        

    @abstractmethod
    def get_start_board(self) -> Tuple[int]: pass


    def done(self) -> int:
        # negative (x) moves counterclockwise (to the right/increasing indices)
        # positive (x) moves clockwise (to the left/decreasing indices)

        # x's home is at the end of the board
        x_home = list(self.board)[-self.homesize:] 
        x_sign = BackgammonEnv.get_sign['x']
        num_x_in_home = abs(sum([x for x in x_home if x < 0]))

        o_home = list(self.board)[:self.homesize]
        o_sign = BackgammonEnv.get_sign['o']
        num_o_in_home = sum([o for o in o_home if o > 0])

        assert not (num_o_in_home == self.ntokens and num_x_in_home == self.ntokens)

        if num_x_in_home == self.ntokens: return x_sign
        if num_o_in_home == self.ntokens: return o_sign

        return 0


    def roll(self) -> List[int]: 
        dice = list(np.random.choice(self.die_nsides, self.ndice) + self.ndice*[1])
        if self.ndice > 1 and all([d == dice[0] for d in dice]):
            dice = [*dice, *dice]
        return dice


    def can_move_from(self, pos: int) -> bool:
        return self.board[pos] and \
            same_sign(BackgammonEnv.get_sign[self.turn], self.board[pos])


    def can_move_to(self, pos: int) -> bool:
        return self.board[pos] == 0 or \
            BackgammonEnv.get_sign[self.turn] == math.copysign(1, self.board[pos]) or \
                (BackgammonEnv.get_sign[self.turn] == -1 * math.copysign(1, self.board[pos]) and abs(self.board[pos]) == 1)


    def verify_action(self, roll: int, from_pos: int, to_pos: int) -> None: 
        # while the player has tokens off the board, they must try to roll in
        if self.out[self.turn] or from_pos is None:
            if self.turn == 'x':
                if to_pos != roll - 1:
                    raise MoveError("Must roll in before moving!")
            else:
                if to_pos != self.size - roll:
                    raise MoveError("Must roll in before moving!")
            # cannot roll in on a position where the opponent has > 1 token
            if not self.can_move_to(to_pos):
                raise MoveError("Cannot roll in on {}".format(to_pos))
            return
        
        # player must select a valid position on the board
        if from_pos < 0: 
            raise MoveError("Position {} is not on the board".format(from_pos))
        if to_pos > self.size - 1:
            raise MoveError("Position {} is not on the board".format(to_pos))

        # player must have a token at the start position
        if not self.can_move_from(from_pos) :
            raise MoveError("Player {} has no tokens at position {}".format(self.turn, from_pos))

        # player can only move to locations with at most one enemy token
        if not self.can_move_to(to_pos):
            raise MoveError("Cannot move {} tokens to Player {}'s position ({})".format(self.turn, 'x' if self.turn == 'o' else 'o', to_pos))
        
        # x/self moves clockwise/increasing
        # make sure the moves add up
        if self.turn == 'x' and (from_pos + roll) != to_pos:
            raise MoveError("Player {} must move clockwise/increasing and from_pos + roll == to_pos".format(self.turn))
        elif self.turn == 'o' and (from_pos - roll) != to_pos:
            raise MoveError("Player {} must move counter-clockwise/decreasing and from_pos - roll == to_pos".format(self.turn))


    def apply_action(self, roll, from_pos, to_pos) -> None:
        sign = BackgammonEnv.get_sign[self.turn]
        board = list(self.board)
        
        # Increment destination
        if board[to_pos] and sign != math.copysign(1, board[to_pos]):
            # knock out the opponent's token
            self.out['x' if self.turn == 'o' else 'o'] += 1
            board[to_pos] = sign
        else: 
            board[to_pos] += sign

        # Decrement source 
        if self.out[self.turn]:
            self.out[self.turn] -= 1
        else:
            board[from_pos] -= sign
        
        self.board = tuple(board)


    def verify_roll(self, roll) -> None:
        if roll not in self.dice:
            raise MoveError("{} not in {}".format(roll, self.dice))

    
    def verify_numtokens(self, roll, from_pos, to_pos) -> None:
        sign_x = BackgammonEnv.get_sign['x']
        sign_o = BackgammonEnv.get_sign['o']
        total_x = sum(abs(i) for i in self.board if math.copysign(1, i) == sign_x) + self.out['x']
        total_o = sum(abs(i) for i in self.board if math.copysign(1, i) == sign_o) + self.out['o']
        if total_x != self.ntokens or total_o !=self.ntokens:
            raise MoveError("Moving {} from {} to {} for roll {} increased total number of tokens for {}".format(self.turn, from_pos, to_pos, roll, self.turn))


    def next_turn(self):
        self.turn = 'x' if self.turn == 'o' else 'o'
        self.dice = self.roll()

    
    def rollin2action(self, action):
        return action, None, action - 1 if self.turn == 'x' else self.size - action


    def step(self, action):
        assert sum(self.board) + self.out['x']*BackgammonEnv.get_sign['x'] + self.out['o']*BackgammonEnv.get_sign['o'] == 0, \
            str(sum(self.board)) + ' ' + str(self.out)

        if action is None or action == 'None': 
            self.next_turn()
            return

        if type(action) is int:
            action = self.rollin2action(action)

        roll = action[0]

        self.verify_roll(roll)      
        self.verify_action(*action)
        self.apply_action(*action)

        self.dice.remove(roll)

        self.verify_numtokens(*action)
        
        if not self.dice: self.next_turn()

        

    def __str__(self):
        # negative moves counterclockwise (to the right/increasing indices)
        # positive moves clockwise (to the left/decreasing indices)
        retval = "Player: {}, Roll: {}\nOut: {}\n\n".format(self.turn, self.dice, self.out[self.turn])
        get_line = lambda cells: "  ".join(["o" if c > 0 else "x" if c < 0 else " " for c in cells])

        x_neg_home = list(self.board)[:self.size//2]
        o_pos_home = list(self.board)[self.size//2:]
        x_neg_home.reverse()
        
        x_neg_indices = [str(i).zfill(2) for i in range(0, self.size//2)]
        x_neg_indices.reverse()
        o_pos_indices = [str(i).zfill(2) for i in range(self.size//2, self.size)]

        retval += " ".join([str(i) for i in o_pos_indices]) + "\n"
        
        while True:
            retval += get_line(o_pos_home) + "\n"
            if sum(abs(np.array(o_pos_home))) == 0: break                
            else: 
                for i, c in enumerate(o_pos_home):
                    if c > 0: o_pos_home[i] -= 1
                    elif c < 0: o_pos_home[i] += 1

        lines_to_rev = []
        while True:
            lines_to_rev.append(get_line(x_neg_home))
            if sum(abs(np.array(x_neg_home))) == 0: break
            else:
                for i, c in enumerate(x_neg_home):
                    if c > 0: x_neg_home[i] -= 1
                    elif c < 0: x_neg_home[i] += 1

        lines_to_rev.reverse()
        return retval + "\n".join(lines_to_rev) +  "\n" + " ".join(x_neg_indices)

 
class MiniGammon5Env(BackgammonEnv):


    def __init__(self):
        BackgammonEnv.__init__(self, size=5, ntokens=2, homesize=1, ndice=1, die_nsides=1)


    def get_start_board(self) -> Tuple[int]:
        board = self.size*[0]

        board[-1] =  2
        board[0]  = -2

        return board

    # def reward(self, action, player_token): pass


# %% [markdown]
# (1 point) Design a reward function for the environment. This can either 
# be a method on the MiniGammon5Env class, or a standalone function.
# Describe your reasoning for the design: did you iterate at all? What did you change, 
# if anything?

# Reflect:
# In the office hour, I noticed that the goal for the agent is not to win the game or gain any advantage on the board but to make the game stuck as soon as possible.
# So, since the agent need to do it ASAP, I assign reward=-1 for each time step, and by default, the reward is 0 when the game stucks.

def reward(state):
    return -1

# %% [markdown]
# Your code will compare the policies of the follow two agents.

class Agent(ABC):
    
    def __init__(self, token):
        self.token = token
        self.value_function = None

    @abstractmethod
    def get_action(self, env: BackgammonEnv) -> Tuple[int, int, int]: pass

    def try_off_policy(self, env, roll, from_pos) -> Union[None, Tuple[int, int, int]]:
        try: 
            if self.token == 'x':
                new_action = (roll, from_pos, roll + from_pos) 
                env.verify_action(*new_action)
                return new_action
            else:
                new_action = (roll, from_pos, from_pos - roll)
                env.verify_action(*new_action)
                return new_action
        except MoveError: return None

    def get_all_valid_actions(self, env):
        all_valid_actions = []
        # Try to roll in, if out
        if env.out[self.token]:
            for roll in env.dice:
                try:
                    action = env.rollin2action(roll)
                    env.verify_action(*action)
                    return [action]
                except MoveError: pass
        # Otherwise, sample from available board positions without replacement
        positions = list(enumerate(env.board))
        for i, count in positions:
            if count and math.copysign(1, count) == math.copysign(1, BackgammonEnv.get_sign[self.token]):
            # if i and math.copysign(1, count) == math.copysign(1, BackgammonEnv.get_sign[self.token]):
                for roll in env.dice:
                    try:    
                        if self.token=='x':
                            action = roll, i, i + roll
                        else:
                            action = roll, i, i - roll
                        env.verify_action(*action)
                        all_valid_actions.append(action)
                    except MoveError: pass
        return all_valid_actions

# # RandomAgent
# This is an agent that takes a random legal action in the environment.

class RandomAgent(Agent): 

    def __init__(self):
        super().__init__('x')

    def get_action(self, env):
        # Try to roll in, if out
        if env.out[self.token]:
            for roll in env.dice:
                try:
                    action = env.rollin2action(roll)
                    env.verify_action(*action)
                    return action
                except MoveError: pass
            return None

        # Otherwise, sample from available board positions without replacement
        positions = list(enumerate(env.board))
        np.random.shuffle(positions)
        for i, count in positions:
            if count and math.copysign(1, count) == math.copysign(1, BackgammonEnv.get_sign[self.token]):
            # if i and math.copysign(1, count) == math.copysign(1, BackgammonEnv.get_sign[self.token]):
                for roll in env.dice:
                    try:    
                        action = roll, i, roll + i
                        # print(f"RandomAgent> possible action {action}")
                        env.verify_action(*action)
                        return action
                    except MoveError: pass
    def get_actions(self,env):
        return self.get_all_valid_actions(env)

# # BasicDeterministicPolicy
# This is an agent that always moves the token that is farthest from the home zone.
class BasicDeterministicPolicy(Agent):

    def __init__(self):
        self.token = 'o'

    def get_action(self, env):
        # Try to roll in, if out
        if env.out[self.token]:
            for roll in env.dice:
                try:
                    action = env.rollin2action(roll)
                    env.verify_action(*action)
                    return action
                except MoveError: pass
            return None

        # try moving the piece that is farthest back first
        positions = list(enumerate(env.board))
        positions.reverse()
        for i, count in positions:
            if math.copysign(1, count) == math.copysign(1, BackgammonEnv.get_sign[self.token]):
                for roll in env.dice:
                    try:
                        action = roll, i, i - roll
                        # print(f"BasicDeterministicPolicy> possible action {action}")
                        env.verify_action(*action)
                        return action
                    except MoveError: pass

    def get_actions(self, env):
        action = self.get_action(env)
        if action is None:
            return []
        else:
            return [action]

class MyPolicy(Agent):
    def __init__(self, agent: Agent):
        self.agent = agent
        self.token = agent.token
        self.policy = {} # should populate the policy lookup table before `get_action()`
    
    def update_policy(self, state, current_actions, next_states, next_values):
        if len(current_actions)==0: # no need to update when there's no valid policy
            best_action = None
        elif len(next_values)==0: # choose the action that cause opponent to have no valid policy
            best_action = current_actions[0]
        else:
            assert len(next_values)==len(current_actions)
            best = np.argmax(next_values)
            best_action = current_actions[best]
        self.policy[state] = best_action

    def get_action(self, env: BackgammonEnv):
        if len(self.policy)==0:
            return None
        return self.policy[tuple(env.board)]
    
    def get_actions(self, env):
        return self.agent.get_actions(env)

def enumerate_states(env: BackgammonEnv) -> List[Tuple[Tuple[int], int]]:
    # (1 point) Enumerate the state space. 
    # You are not required to output the state representation indicated
    # by the type signature, but you may want to consider it. 
    #
    # In the reflection, describe how you chose to represent the state.
    # What pieces of the environment did you include? What pieces of the environment
    # did you exclude? Why?
    all_states = set()

    empty_board = [0,0,0,0,0]
    num_positions = len(empty_board)
    for x,o in [(-1,1), (1,-1)]:
        for i in range(num_positions+1): # additional digit for "out"
            for j in range(num_positions+1):
                board = empty_board.copy()
                if i < num_positions:
                    board[i] += x
                if j < num_positions:
                    board[j] += x
                for i1 in range(num_positions+1):
                    for j1 in range(num_positions+1):
                        board1 = board.copy()
                        if i1 < num_positions:
                            if board1[i1]>0:
                                board1[i1] = o
                            else:
                                board1[i1] += o
                        if j1 < num_positions:
                            if board1[j1]>0:
                                board1[j1] = o
                            else:
                                board1[j1] += o

                        board1 = tuple(board1)
                        all_states.add(board1)

    # print(all_states)
    assert (1,2,3) == (1,2,3)
    print(len(all_states))
    return all_states


def policy_evaluation(states, policy: Agent, env: BackgammonEnv, oppo: Agent, discount: float = 0.1, value_function = None) -> Dict[Tuple[Tuple[int], int], int]:
    # (1 point) This function will compute the value function for your policy.
    # You may choose your own value of the discount factor and threshold for convergence
    # You will need to provide an opposition agent to advance the state through a turn. 
    # The choice of opposition agent is up to you.
    #
    # Arguments
    # ---------
    # policy: the agent whose value function you are computing
    # env: an instance of MiniGammon5
    # oppo: the opposition agent
    #
    # Return
    # ------
    # A map from each state representation to its value
    # Note: you are not required to use the state representation indicated by the 
    # type signature. 
    # 
    # Before you begin, record a description of what you expect the value function to look like.
    # Then run policy_evaluation for both RandomAgent and BasicDeterministicAgent.
    # In the reflection, describe what any behavior you found surprising in the output. 
    # For each agent's value function, what is the value of state when the board is 
    # (0, 0, 2, -2, 0)?
    #
    # Hints and things to watch out for:
    # * The whose value function you are computing does not take 
    #   an action in the environment until after the opponent has played. 
    #   you may want to create an auxiliary function that returns the next state 
    #   at which the policy can act.
    # * The environment is stateful, so you will want to make sure that you are resetting 
    #   the environment appropriately.
    # * You may want to use the `env.set_state` function. Be sure to check its type signature.

    # Reflect:
    # I don't have any expectation before running policy evaluation. Any thing should be possible.
    # But there is still surprise: the value all end up close to -1.1.
    # I thought about it for a while, and that makes sense, because the discount factor is 0.1, and the reward is -1, so the value function is approximately -1 + 0.1 * (-1) if the game doesn't end.
    # I think the discount factor is a little small in practice.
    # I am not sure why you ask a state that is not reachable. The start state is [-2,0,0,0,2]. They won't switch sides.
    # Assuming we don't consider whether it is reachable (in practice, we usually don't know if a state is reachable or not unless intentionally optimize for it.)
    # Value is -1.1 for either agent. It makes sense because the game is going to end in next two steps. 
    # and the agent won't have any choice other than moving one and moving the other one, and game end.

    game = env
    if value_function is None:
        value_function = {}
        for state in states:
            value_function[state] = -999
    # in-place policy evalution (Page 75)
    step = 0
    while True:
        step += 1
        delta = 0
        for state in states:
            old_value = value_function[state]
            # print(f"state = {state}")
            game.set_state(state, [1], turn=policy.token)
            done = False
            new_value = 0
            possible_actions = policy.get_actions(game)
            if len(possible_actions)==0:
                done = True
            for action in possible_actions:
                game.set_state(state, [1], turn=policy.token)
                game.step(action)
                tmp_state = game.board
                assert oppo.token == game.turn

                oppo_possible_actions = oppo.get_actions(game)
                if len(oppo_possible_actions)==0:
                    done = True
                for oppo_action in oppo_possible_actions:
                    game.set_state(tmp_state, [1], turn=oppo.token)
                    game.step(oppo_action)
                    next_state = tuple(game.board)
                    # print(f"next_state = {next_state}")
                    new_value += reward(state) + discount * value_function[next_state]
            if done:
                new_value = 0
            else:
                new_value = new_value / (len(possible_actions) * len(oppo_possible_actions))
            if False and new_value!=old_value:
                print(f"Value changed in policy evalution: new {new_value} old {old_value}")
            delta = max(delta, np.abs(new_value - old_value))
            value_function[state] = new_value
        if delta < 0.1:
            break
    print(f"policy_evaluation> done in {step} steps.")
    return value_function


def policy_improvement(states, policy: Agent, env: BackgammonEnv, oppo: Agent, vfn: Dict[Tuple[Tuple[int], int], int], discount : float = 0.1) -> Tuple[Agent, bool]:
    # (1 point) This function will update the policy once.
    # You may choose to update this policy either by wrapping the get_action method or 
    # building an explicit lookup table. 
    # You should make sure you use the same agents as you did for policy evaluation.
    #
    # Arguments
    # ---------
    # policy: the agent whose policy you want to improve
    # env: an instance of MiniGammon5
    # oppo: the opposition agent
    # vfn: the value function under policy, expressed how you see fit (I recommend
    # expressing it as a map of type: Dict[Tuple[Tuple[int], int], int], i.e.,
    # (int * int * int * int * int * int) * int -> int
    #
    # Before you begin, write down your expectations for the behavior of the agent under
    # policy improvement. What states do you expect to be updated? What states do you 
    # expect to remain the same? Are there any features of the state such that you 
    # might know a priori that the policy's behavior is optimal? Do you expect both, 
    # neither, or only one of the RandomAgent and BasicDeterministicAgent's policies to 
    # change? Why or why not?
    # 
    # After you have implemented policy improvement, did you observe and changes to 
    # either input policy at all? If yes, how many states improved? Did they have anything
    # in common?
    #
    # Hints and things to watch out for:
    # * The hints in the policy_evaluation section.
    # * You may want to use `policy.try_off_policy` to get the off-policy action (i.e,
    #   the action whose q-value you are computing, that is not equal to the action taken 
    #   when we just follow the policy).

    # Reflect:
    # My expectation was that the policy will update some mappings to produce better actions at some states.
    # I expected some of the non-terminating states will be updated.
    # Terminating states will not be updated.
    # This optimization is based on the value function calculated, so I can check the value function and manually compute which action is optimal.
    # I expected both policies would change. Because this process is based on the value function, and we newly computed the value function, and it's not possible that those original policies will result the right actions because they don't know the value function.
    # After implemented this function, I observed that the policy changed in either case.
    # 20 states were improved in the first run for the RandomAgent.
    # 44 states were improved in the first run for the BasicDeterministicPolicy.
    # They both improved a lot.

    game = env
    new_policy = MyPolicy(policy)
    for current_state in states:
        game.set_state(current_state, [1], turn=policy.token)
        valid_actions = policy.get_all_valid_actions(game)
        current_actions = []
        next_states = []
        next_values = []
        opponent_have_no_valid_action = False
        for action in valid_actions:
            game.set_state(current_state, [1], turn=policy.token)
            game.step(action)
            tmp_state = game.board
            assert oppo.token == game.turn

            oppo_valid_actions = oppo.get_all_valid_actions(game)
            if len(oppo_valid_actions)==0: # great, our opponent have no valid action!
                current_actions = [action]
                opponent_have_no_valid_action = True
                break
            for oppo_action in oppo_valid_actions:
                game.set_state(tmp_state, [1], turn=oppo.token)
                game.step(oppo_action)
                next_state = tuple(game.board)
                current_actions.append(action)
                next_states.append(next_state)
        if opponent_have_no_valid_action:
            next_states = []
            next_values = []
        else:
            for next_state in next_states:
                next_values.append(vfn[next_state])
        new_policy.update_policy(current_state, current_actions, next_states, next_values)

    state_updated = 0
    for state in states:
        # for the reflection question:
        game.set_state(state, [1], turn=policy.token)
        old_action = policy.get_action(game)
        game.set_state(state, [1], turn=policy.token)
        new_action = new_policy.get_action(game)
        if old_action!=new_action:
            state_updated += 1
    print(f"policy_improvement> Improved {state_updated} states.")

    return new_policy

def policy_iteration(states, policy: Agent, env: BackgammonEnv, oppo: Agent, vfn: Any, discount: float=0.1) -> Tuple[Agent, Dict[Tuple[Tuple[int], int], int]]:
    # (1 point) Combine policy evaluation and policy improvement to produce a new agent.
    # 
    # How many passes of policy iteration did you run before your value function converged?
    # (Answer for both RandomAgent and BasicDeterministicAgent)

    # Reflection:
    # For RandomAgent, the policy iteration looped once (22 states were improved) and the second iteration it was already converged.
    # Same for BasicDeterministicPolicy, the policy iteration looped once (44 states were improved) and the second iteration it was already converged.
     
    game = env
    iteration = 0
    while True:
        iteration += 1
        print(f"policy_iteration> Iter {iteration}")
        value_function = policy_evaluation(states, policy, env, oppo, discount, value_function=vfn)
        new_policy = policy_improvement(states, policy, env, oppo, value_function, discount)
        if policy.token=='x':
            print(f"policy_iteration> value(-2,0,0,2,0): {value_function[(-2,0,0,2,0)]}")
        policy_stable = True
        for state in states:
            game.set_state(state, [1], turn=policy.token)
            old_action = policy.get_action(game)
            game.set_state(state, [1], turn=policy.token)
            new_action = new_policy.get_action(game)
            if old_action != new_action:
                policy_stable = False
        policy = new_policy
        if policy_stable:
            break
    print(f"policy_iteration> loops {iteration} times.")
    return policy

def value_iteration(states, policy: Agent, env: BackgammonEnv, oppo: Agent, discount: float = 0.1) -> Tuple[Agent, Dict[Tuple[Tuple[int], int], int]]:
    # (1 point) Implement value iteration.
    #
    # How many passes of value interation did you run before your value function converged?
    # (Answer for both RandomAgent and BasicDeterministicAgent)

    # Reflect:
    # For RandomAgent, the value iteration looped 7 times and converged at the 8-th iteration (28 states were improved).
    # For BasicDeterministicAgent, the value iteration looped 9 times and converged at the 10-th iteration (50 states were improved).
    value_function = {}
    for state in states:
        value_function[state] = -999
    # Page 83
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for state in states:
            old_value = value_function[state]
            # print(f"state = {state}")
            game.set_state(state, [1], turn=policy.token)
            done = False
            possible_actions = policy.get_actions(game)
            possible_action_values = []
            if len(possible_actions)==0:
                new_value = 0
            else:
                for action in possible_actions:
                    game.set_state(state, [1], turn=policy.token)
                    game.step(action)
                    tmp_state = game.board
                    assert oppo.token == game.turn

                    oppo_possible_actions = oppo.get_actions(game)
                    if len(oppo_possible_actions)==0:
                        done = True

                    action_value = 0
                    for oppo_action in oppo_possible_actions:
                        game.set_state(tmp_state, [1], turn=oppo.token)
                        game.step(oppo_action)
                        next_state = tuple(game.board)
                        # print(f"next_state = {next_state}")
                        action_value += reward(state) + discount * value_function[next_state]
                    if done:
                        action_value = 0
                    else:
                        action_value = action_value / len(oppo_possible_actions)
                    
                    possible_action_values.append(action_value)
                new_value = np.max(possible_action_values)
            delta = max(delta, np.abs(new_value - old_value))
            value_function[state] = new_value
        # if policy.token=='x':
        #     print(f"value_iteration> value(-2,0,0,2,0): {value_function[(-2,0,0,2,0)]}")
        if delta<0.1:
            break
    print(f"value_iteration> loops {iteration} times.")

    new_policy = policy_improvement(states, policy, env, oppo, value_function, discount)
    return new_policy, value_function
    
    
# (1 point -- EXTRA CREDIT) Run policy iteration for your agents from
# PA1. (You will need to modify those agents to work in the MiniGammon5 environment and to 
# conform to the Agent interface). 
#
# How close were your agents to optimal (e.g., how many updates were made)? How far off from 
# the optimal policy produced by value_iteration were your agents?

# %%
if __name__ == "__main__":
    import argparse
    # looks like argparse strips formatting?
    parser = argparse.ArgumentParser(description="""
    You can interact with the game, playing as both players. Each time step is a move. Each turn is comprised of 1-4 moves. 
    The orientation of the printed board is such that the lower right is the "home zone" for Player o and the upper right is the "home zone" for Player x. 
    There are three different types of actions:

    1. Regular action (x, y, x): a tuple of one of the rolls of the dice (x), the location of the token you wish to pick up (y), and the location you wish to move to (z).

    2. Roll in: when the opponent of the current player "knocks out" one of your tokens, you will need to roll in. You can provide a single number corresponding to the face of the die you wish to roll in on. This will place your token at the die's offset from the extreme point in the players home. 

    3. Skip: you can hit "enter" in interactive mode, or use "None" in moves file to skip your turn (you must do this when there are no valid moves; the game will not provide this information for you).
    """)
    parser.add_argument('--move_file', help="A list of actions delimtied by newline")
    parser.add_argument('--player1')
    parser.add_argument('--player2')

    args = parser.parse_args()

    #game = BackgammonEnv(None)
    game = MiniGammon5Env()

    # if args.move_file:
    #     with open(args.move_file) as f:
    #         for move in f.readlines():
    #             game.step(eval(move))
    #             print(game)

    player1, player2 = None, None
    if args.player1:
        player1 = eval(args.player1)()
    if args.player2:
        player2 = eval(args.player2)() 

    states = enumerate_states(game)
    print('{} states total'.format(len(states)))
    print("===== Policy Evaluation =====")
    v1 = policy_evaluation(states, RandomAgent(), game, BasicDeterministicPolicy())
    v2 = policy_evaluation(states, BasicDeterministicPolicy(), game, RandomAgent())
    print(game.get_start_board())
    print(v1[tuple(game.get_start_board())])
    print(v2[tuple(game.get_start_board())])
    print(v1[((0, 0, 2, -2, 0))])
    print(v2[((0, 0, 2, -2, 0))])
    # see states where the value functions for each policy differ:
    # for s in states:
    #     if v1[s] != v2[s]:
    #         print(s, v1[s], v2[s])
    print('===== Policy Improvement ====')
    print("")
    pi1 = policy_improvement(states, RandomAgent(), game, BasicDeterministicPolicy(), v1)
    print("")
    pi2 = policy_improvement(states, BasicDeterministicPolicy(), game, RandomAgent(), v2)
    print("")
    print('===== Policy iteration ====')
    print("")
    pi1_ = policy_iteration(states, RandomAgent(), game, BasicDeterministicPolicy(), v1)
    print("")
    pi2_ = policy_iteration(states, BasicDeterministicPolicy(), game, RandomAgent(), v2)
    print("")
    print('==== Value iteration =====')
    print("")
    player1, v1_vi = value_iteration(states, RandomAgent(), game, BasicDeterministicPolicy())
    print("")
    player2, v2_vi = value_iteration(states, BasicDeterministicPolicy(), game, RandomAgent())
    print("")
    print('==== Debug =====')
    print("")
    policy_iteration(states, player1, game, BasicDeterministicPolicy(), v1_vi)
    policy_iteration(states, player2, game, RandomAgent(), v2_vi)

    # # Demo of the agents playing the game
    # # Run this with `python assignemnt2_soln.py --player1 RandomAgent --player2 BasicDeterministicPolicy`
    # print('player1 ({}) is {}'.format(player1.token, player1.__class__.__name__))
    # print('player2 ({}) is {}'.format(player2.token, player2.__class__.__name__))
    # numsteps = 0
    # # reset the env
    # game.set_state(game.get_start_board(), game.roll())
    # done = game.done()
    # # uncomment the below to alias players with results from value iteration

    # while not done:
    #     try:
    #         print("\n===================================\n" + str(game) + "\n===================================\n")
    #         if player1 and game.turn == player1.token:
    #             action = player1.get_action(game)
    #         elif player2 and game.turn == player2.token:
    #             action = player2.get_action(game)
    #         else:
    #             if game.out[game.turn]: 
    #                 print("First move must roll in (list roll)")
    #             print("Move {} (format (roll, from, to))>".format(game.turn), end=' ')
    #             action = eval(input() or "None")
    #         if action==None:
    #             break
    #         game.step(action)
    #         numsteps += 1
    #     except MoveError as e:
    #         print('------------------------------------------------------------------------------------------')
    #         print("ERROR: " + str(e))
    #         print('------------------------------------------------------------------------------------------')
    #     done = game.done()

    # if done > 0:
    #     print('Player o won in {} moves!'.format(numsteps))
    # elif done < 0:
    #     print('Player x won in {} moves!'.format(numsteps))
    # else:
    #     print('Game Stuck.')
    

    import pickle
    results = [
        pi1, pi2, pi1_, pi2_, player1, player2, v1, v2, v1_vi, v2_vi,
    ]
    with open("./sida.pickle", "wb") as f:
        pickle.dump(results, f)