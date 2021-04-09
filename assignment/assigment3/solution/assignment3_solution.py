import math
import numpy as np
# You may want to uncomment the setting of the seed, to test.
#np.random.seed(1234)

from abc import abstractmethod, ABC
from collections import Counter
from typing import Tuple, List, Dict, Any, Union

#%% [markdown]
# # Programming Asignment 3
# The goal of this second programming assignment is to give you experience sampling from trajectories
# in order to learn value functions and build better policies. You will again be using the MiniGammon5
# environment, but this time with a 2-sided die.
#
# Tips and Hints:
# * Some of these functions will take a long time to terminate! Try to test incrementally, make generous
#   use of print statements, and consider treating each part of the assginment separately. 
# * Keep a notebook of what you've tried as you go. 

#%%
Board = Tuple[int]
State = Tuple[Board, int, str]
Action = Union[Tuple[int, Union[int, None], int], None]

class MoveError(Exception): pass

def same_sign(a, b):
    return math.copysign(1, a) == math.copysign(1, b)


class BackgammonEnv():

    get_sign = {'x' : -1, 'o': 1}    
    get_token = {-1: 'x', 1: 'o'}
    default_start = 'x'


    def __init__(self, size=24, ntokens=15, homesize=6, ndice=2, die_nsides=6):
        # configuration information
        self.size = size
        self.ntokens = ntokens
        self.homesize = homesize
        self.ndice = ndice
        self.die_nsides = die_nsides

        # moved board, roll, and turn token into state object
        self.board = self.get_start_board()
        self.dice = self.roll()
        self.turn = BackgammonEnv.default_start
        # made 'out' a function, since it can be computed from state

        # Using this to check for more complex loops in gameplay
        self.states = set()

    
    def out(self):
        board = self.board
        pos = sum([t for t in board if t > 0])
        neg = abs(sum([t for t in board if t < 0]))
        return {'x' : self.ntokens - neg, 'o': self.ntokens - pos}


    def set_state(self, board: Tuple[int], dice: List[int], turn=None) -> None:
        self.board = board
        self.dice = dice
        self.turn = turn or self.turn
        self.states = set()
        

    @abstractmethod
    def get_start_board(self) -> Tuple[int]: pass


    def get_state(self):
        """Returns a hashable state"""
        return (tuple(self.board), tuple(self.dice), self.turn)


    def get_direction(self, token):
        # negative (x) moves counterclockwise (to the right/increasing indices)
        # positive (x) moves clockwise (to the left/decreasing indices)
        return BackgammonEnv.get_sign[token] * -1


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

        assert not (num_o_in_home == self.ntokens and num_x_in_home == self.ntokens), \
            'Cannot tie! Are you missing an env.done() earlier?'

        if num_x_in_home == self.ntokens: return x_sign
        if num_o_in_home == self.ntokens: return o_sign
        
        return 0
        

    def roll(self) -> List[int]: 
        dice = list(np.random.choice(self.die_nsides, self.ndice) + self.ndice*[1])
        if self.ndice > 1 and all([d == dice[0] for d in dice]):
            dice = [*dice, *dice]
        return [int(d) for d in dice] # ensures that they are all Python ints rather than numpy ints


    def can_move_from(self, pos: int) -> bool:
        return bool(self.board[pos]) and \
            same_sign(BackgammonEnv.get_sign[self.turn], self.board[pos])


    def can_move_to(self, pos: int) -> bool:
        board  = self.board
        turn = self.turn
        return board[pos] == 0 or \
            BackgammonEnv.get_sign[turn] == math.copysign(1, board[pos]) or \
                (BackgammonEnv.get_sign[turn] == -1 * math.copysign(1, board[pos]) and abs(board[pos]) == 1)


    def verify_action(self, roll: int, from_pos: int, to_pos: int, turn=None) -> None: 
        if self.done(): raise MoveError("Cannot move after terminal state.")
        turn = turn or self.turn
        # while the player has tokens off the board, they must try to roll in
        if self.out()[turn]:
            if turn == 'x':
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
        if turn == 'x' and (from_pos + roll) != to_pos:
            raise MoveError("Player {} must move clockwise/increasing and from_pos + roll == to_pos".format(turn))
        elif turn == 'o' and (from_pos - roll) != to_pos:
            raise MoveError("Player {} must move counter-clockwise/decreasing and from_pos - roll == to_pos".format(turn))


    def apply_action(self, roll, from_pos, to_pos) -> None:
        sign = BackgammonEnv.get_sign[self.turn]
        board = list(self.board)
        
        # Increment destination
        if board[to_pos] and sign != math.copysign(1, board[to_pos]):
            # knock out the opponent's token
            self.out()['x' if self.turn == 'o' else 'o'] += 1
            board[to_pos] = sign
        else: 
            board[to_pos] += sign

        # Decrement source 
        if self.out()[self.turn]:
            self.out()[self.turn] -= 1
        else:
            board[from_pos] -= sign
        
        self.board = tuple(board)


    def verify_roll(self, roll) -> None:
        if roll not in self.dice:
            raise MoveError("{} not in {}".format(roll, self.dice))

    
    def verify_numtokens(self, roll, from_pos, to_pos) -> None:
        sign_x = BackgammonEnv.get_sign['x']
        sign_o = BackgammonEnv.get_sign['o']
        total_x = sum(abs(i) for i in self.board if math.copysign(1, i) == sign_x) + self.out()['x']
        total_o = sum(abs(i) for i in self.board if math.copysign(1, i) == sign_o) + self.out()['o']
        if total_x != self.ntokens or total_o !=self.ntokens:
            raise MoveError("Moving {} from {} to {} for roll {} increased total number of tokens for {}".format(self.turn, from_pos, to_pos, roll, self.turn))


    def next_turn(self):
        self.turn = 'x' if self.turn == 'o' else 'o'
        self.dice = self.roll()

    
    def rollin2action(self, action: Any) -> Tuple[int, Union[int, None], int]:
        return action, None, action - 1 if self.turn == 'x' else self.size - action


    def step(self, action: Action):
        assert sum(self.board) + self.out()['x']*BackgammonEnv.get_sign['x'] + self.out()['o']*BackgammonEnv.get_sign['o'] == 0, \
            str(sum(self.board)) + ' ' + str(self.out())

        if action is None or action == 'None': 
            self.next_turn()
            return

        # if type(action) is int:
        #     action: Tuple[int, int, int] = self.rollin2action(action)

        roll = action[0]

        self.verify_roll(roll)      
        self.verify_action(*action)
        self.apply_action(*action)

        self.dice.remove(roll)

        self.verify_numtokens(*action)
        
        if not self.dice: self.next_turn()


    def reset(self):
        self.set_state(self.get_start_board(), self.roll(), turn=BackgammonEnv.default_start)
        

    def __str__(self):
        # negative moves counterclockwise (to the right/increasing indices)
        # positive moves clockwise (to the left/decreasing indices)
        retval = "Player: {}, Roll: {}\nOut: {}\n\n".format(self.turn, self.dice, self.out()[self.turn])
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



class FullBackammon(BackgammonEnv):

    def get_start_board(self):
        board = self.size*[0]

        board[-1] = 2
        board[12] = 5
        board[7]  = 3
        board[5]  = 5
        board[0]   = -2
        board[-13] = -5
        board[-8]  = -3
        board[-6]  = -5

        return board


class MiniGammon12Env(BackgammonEnv):

    def __init__(self):
        BackgammonEnv.__init__(self, size=12, ntokens=7, homesize=3, ndice=1)

    def get_start_board(self):
        board = self.size*[0]

        board[-1] =  2
        board[5]  =  5
        board[0]  = -2
        board[-6] = -5

        return board

class MiniGammon8Env(BackgammonEnv):
    
    def __init__(self):
        BackgammonEnv.__init__(self, size=8, ntokens=5, homesize=2, ndice=1)

    def get_start_board(self):
        board = self.size*[0]

        board[-1] = 2
        board[5] = 3
        board[0] = -2
        board[-5] = -3

        return board
 

class MiniGammon5Env(BackgammonEnv):


    def __init__(self):
        BackgammonEnv.__init__(self, size=5, ntokens=2, homesize=1, ndice=1, die_nsides=2)


    def get_start_board(self) -> Tuple[int]:
        board = self.size*[0]

        board[-1] =  2
        board[0]  = -2

        return board


    def reward(self, action, player_token):
        player_sign = BackgammonEnv.get_sign[player_token]

        if player_sign == -1: 
            if self.board[-1] == -2:
                # This player has won. :)
                return 0
            elif self.board[0] == 2: 
                # The opposition player has won. :(
                return -2

        if player_sign == 1:
            if self.board[0] == 2: 
                # This player has won. :)
                return 0
            elif self.board[-1] == -2: 
                # The opposition player has won. :(
                return -2

        # getting stuck is very bad
        if action is None: return -10

        return -1

# %% [markdown]
# # Agents 
# This section contains the base class and several agent instantiations.
# %%

class Agent(ABC):
    
    def __init__(self, token):
        self.token = token

    @abstractmethod
    def get_action(self, env: BackgammonEnv) -> Tuple[Action, float]: pass

    @staticmethod
    def get_possible_from_pos(env: BackgammonEnv, token: str) -> List[int]:
        """Returns indices of positions from which the player can move. 
        
        Includes endpoints, since these will be filtered out by verify_action."""
        return [i for (i, ct) in enumerate(env.board) if \
            ct and \
            (math.copysign(1, ct) == BackgammonEnv.get_sign[token])]

    @staticmethod
    def try_roll_in(env) -> Tuple[Action, float]:
        action_set_size = 0
        action_to_take = None
        # Note: we don't need to shuffle env.dice because they are already the 
        # result of a random process
        for roll in env.dice:
            try:
                action = env.rollin2action(roll)
                env.verify_action(*action)
                action_set_size += 1
                action_to_take = action
            except MoveError: pass
        return (action_to_take, 1 / float(action_set_size or 1))


class RandomAgent(Agent): 

    def __init__(self, token='x'):
        self.token = token

    def get_action(self, env) -> Tuple[Action, float]:
        # Try to roll in, if out
        if env.out()[self.token]: return self.try_roll_in(env)

        # Otherwise, sample from available board positions without replacement
        positions = Agent.get_possible_from_pos(env, self.token)
        np.random.shuffle(positions)

        action_set_size = 0
        action_to_take = None
        for i in positions:
            for roll in env.dice:
                try:    
                    action = roll, i, i + (roll * env.get_direction(self.token))
                    env.verify_action(*action)
                    action_set_size +=1
                    action_to_take = action
                except MoveError: pass
        return (action_to_take, 1 / float(action_set_size or 1))


class BasicDeterministicPolicy(Agent):

    def __init__(self, token='o'):
        self.token = token

    def get_action(self, env) -> Tuple[Action, float]:
        # Try to roll in, if out
        if env.out()[self.token]:
            for roll in env.dice:
                try:
                    action = env.rollin2action(roll)
                    env.verify_action(*action)
                    return (action, 1.0)
                except MoveError: pass
            return (None, 1.0)

        # try moving the piece that is farthest back first
        positions = Agent.get_possible_from_pos(env, self.token)
        positions.reverse()
        for i in positions:
            for roll in env.dice:
                try:
                    action = roll, i, i + (roll * env.get_direction(self.token))
                    env.verify_action(*action)
                    return (action, 1.0)
                except MoveError: pass
        return (None, 1.0)



class SoftDeterministicAgent(BasicDeterministicPolicy):

    def __init__(self, eps=0.1, token='o'):
        self.eps = eps
        super().__init__(token=token)

    def get_action(self, env) -> Tuple[Action, float]:
        # Try to roll in, if out
        if env.out()[self.token]: return self.try_roll_in(env)
        
        # take a random action
        draw = np.random.random()
        if draw < self.eps:
            positions = Agent.get_possible_from_pos(env, self.token)
            action_set_size = 0
            action_to_take = None
            np.random.shuffle(positions)
            for i in positions:
                for roll in env.dice:
                    try:
                        action = roll, i, i + (roll * BackgammonEnv.get_sign[self.token] * -1)
                        env.verify_action(*action)
                        action_set_size += 1
                        action_to_take = action
                    except MoveError as e: pass
            return (action_to_take, self.eps * 1/float(action_set_size or 1))

        # otherwise, take the usual action
        action, p = super().get_action(env)
        return (action, (1-self.eps) * p)


# %% [markdown]
# # Utility functions for the assignment.

def enumerate_states(board, states, sign):
    env = MiniGammon5Env()
    token = BackgammonEnv.get_token[sign]
    no_move = True

    # Shouldn't have to check env.done, since env should return the same state when done

    for roll in range(1, env.die_nsides + 1):
        env.set_state(board, [roll], turn=token)
        s = (tuple(board), roll, token) 
        if s in states: continue
        states.add(s)

        # If the current player has tokens out, they need to roll in. 
        # Since there is only one die, they don't have any options for rolling in. 
        if env.out()[token]:
            to_pos = len(board) - roll if sign > 0 else roll - 1
            action = (roll, None, to_pos)
            try:
                env.verify_action(*action, turn=token)
                no_move = False
                env.step(action)
                enumerate_states(env.board, states, (sign * -1))
            except MoveError: pass

        else:
            for i in Agent.get_possible_from_pos(env, token):
                env.set_state(board, [roll], turn=token)
                action = (roll, i, i + (roll * env.get_direction(token)))
                try:
                    env.verify_action(*action, turn=token)
                    env.step(action)
                    no_move = False
                    enumerate_states(env.board, states, (sign * -1))
                except MoveError: pass
    
    if no_move:
        s = (tuple(board), None, token) 
        if s not in states:
            env.step(None)
            states.add(s)
            enumerate_states(env.board, states, (sign * -1))

    return states


def test_trajectory(env, p1, p2):
    for i in range(1000):
        env.reset()
        if i % 100 == 0: print('{}th iteration...'.format(i))
        t = get_trajectory(env, p1, p2)
        last_reward = t[-1][2]
        # we either won or we lost, or had a drawassign 
        # just can't stop if there are more moves
        assert last_reward != -1, 'Reward: {}'.format(last_reward) 


def get_action_probility_for_state(s: State, a: Action, pi: Agent) -> float:
    # returns pi(a|s)
    # agents' get_action methods need an env argument
    env = MiniGammon5Env()
    while True:
        board, roll, turn = s
        env.set_state(board, [roll], turn=turn)
        actual, p = pi.get_action(env)
        if actual == a: return p

# %% [markdown]
# This is the actual assignment portion.

def get_trajectory(env: MiniGammon5Env, p1: Agent, p2: Agent) -> List[Tuple[State, Action, int]]:
    # (1 point) Get a sample trajectory. 
    #
    # Hints:
    # * Make sure you include the final reward (remember the indexing of trajectories).
    # * When stepping through a time unit, make sure you check whether the first player has 
    #   has won before having the second player take their turn.
    retval = []
    env.reset()
    done = env.done()

    while not done:

        s = (tuple(env.board), env.dice[0])
        a, _ = p1.get_action(env)
        r = env.reward(a, p1.token)
        retval.append((s, a, r))

        env.step(a)
        done = env.done()
        if done: break
        env.step(p2.get_action(env)[0])
        done = env.done()

    # Add the final reward
    s = (tuple(env.board), env.dice[0])
    a = None
    r = env.reward(a, p1.token)
    retval.append((s, a, r))

    return retval


def on_policy_exploring_starts(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent, states: List[Tuple[Board, int, str]]) -> Dict[State, float]:
    # (1 point) Compute the value function of the RandomAgent with exploring starts
    # Compute the average returns for 30 sample trajectories.
    # In the reflection, report the value of the arbitrarily chosen state ((1, 1, 0, 0, -1), 1, 'x'), for the RandomAgent.
    #
    # Hints:
    # * You may be tempted to re-use trajectories; you do not need to. Feel free to do independent draws for each state.
    
    # This is a naive implementation that doesn't leverage the fact that some states appear in the trajectories we will be generating.
    vs = {}
    for s in states: 
        board, roll, turn = s
        env.set_state(board, [roll], turn=turn)
        sample = []
        for _ in range(30):
            env.set_state(board, [roll], turn=turn)
            sample.append(get_trajectory(env, p1, p2))
        g = np.average([sum([r for (_, _, r) in t]) for t in sample])
        vs[s] = g

    return vs



def on_policy_soft(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent, min_obs=0):
    # (1 point) Compute the value function of the RandomAgent by allowing it to explore. 
    # 
    # In the reflection, answer the questions: How many sample trajectories did you take in order to converge?
    # What is the smallest number of observations you encountered for your set of state-action pairs?
    #
    # Set a minimum number of observations. What number did you choose? Why?
    #
    # Now update the function to only return if it has seen at least this number of state-action pairs *and*
    # has converged. How many sample trajectories did you draw? Given infinite samples, soft policies should 
    # take every possible action for each state they encounter. Did your agent fail to encouter any states?
    #
    # Hints
    # * You may find it easier to implement this function using the approach described in the lecture video,
    #   which is based on the bandits videos, rather than the algorithm given in the book (which )
    # * Start without requiring a minimum number of visits, and make sure that works before adding in the code
    #   that ensures you've seen the minimum number of observations. Increase the number of minimum required
    #   number of observations slowly.
    # * You will want generous print states, some of which may look something like this:
    # * You may want to use a multiplicative factor to determine your stopping condition, e.g. 
    #   abs(v_old/v_new - 1) < theta, rather than abs(v_old - v_new) < theta
    #
    # if num_samples % 5000 == 0: 
    #     print(num_samples, 'episode length', len(trajectory), 'start state', trajectory[0], 'final state', trajectory[-1])
    #
    # if num_samples % 1000 == 0 and delta > 0.1: 
    #     print(num_samples, '{}/{}'.format(i+1, len(trajectory)), t, '\t v: {}  v\':{}  ct:{}'.format(v_old, v, n_old, g), '\t', delta)


    vs = {}
    ct = {}
    updated = True
    num_samples = 0
    
    while updated:
        updated = False
        num_samples += 1
        trajectory = get_trajectory(env, p1, p2)
        visited = set()

        for i, t in enumerate(trajectory):
            s, _, _ = t
            # numpy ints won't hash to the same place as built-in ints
            if s in visited: continue
            visited.add(s)

            v_old = vs[s] if s in vs else 0
            g = sum([r for (_, _, r) in trajectory[i:]])
            n_old = ct[s] if s in ct else 0
            v = (v_old * n_old + g) / (n_old + 1)
            delta = abs(v/v_old - 1) if v_old else abs(v_old - v)

            if num_samples % 1000 == 0 and delta > 0.1: 
                print(num_samples, '{}/{}'.format(i+1, len(trajectory)), t, '\t v: {}  v\':{}  ct:{}'.format(v_old, v, n_old, g), '\t', delta)
                
            
            if delta > 0.1:
                vs[s] = v
                ct[s] = n_old + 1
                updated = True

        # Convergence for Random agent.
        # no minimum required samples: 98 samples, observed 234 states
        # basically anything higher than that then starts to get into low-frequency states that take forever 
        # to converge. I haven't cared to wait more than an hour but maybe someone else will. 
        if any([n < min_obs for n in ct.values()]): updated = True

    print('Converged with {} samples; observed {} states'.format(num_samples, len(vs)))
                
    return vs


def importance_sampling(env: BackgammonEnv, pi: SoftDeterministicAgent, b: RandomAgent):
    # (1 point) Implement importance sampling for estimating value functions. 
    #
    # Hints:
    # * Your implementation can more or less follow the algorithm in the book/lectures
    # * You will want to play around with your convergence condition, especially if you 
    #   choose not to use a discount factor. I recommend a multiplicative convergence 
    #   condition, similarly to the basic monte carlo sampling you implemented for 
    #   soft policies.

    vs = {}
    ct = {}
    num_samples = 0
    updated = True

    while updated:
        updated = False
        num_samples += 1

        h = get_trajectory(env, b, b)
        g = 0
        w = 1
        visited = set()
        for i, (s, a, _) in enumerate(h):
            if w == 0: break
            if s in visited: break
            visited.add(s)

            g = sum([r for (_, _, r) in h[i+1:]])
            ct[s] += w
            
            v_old = vs[s]
            v_new = (w / ct[s])(g - v_old)
            vs[s] = v_old + v_new

            p_pi = get_action_probility_for_state(s, a, pi)
            p_b = get_action_probility_for_state(s, a, b)
            
            w = w * (p_pi / p_b)
            
            delta = abs(v_new/v_old - 1) if v_old else abs(v_old - v_new)
            
            if delta > 0.1:
                updated = True

    return vs


def off_policy_mc_control(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent):
    # (1 point) Implement off policy monte carlo control, to produce an agent. 
    #
    # Monte Carlo methods are not guaranteed to produce an optimal agent in finite steps. 
    # What percentage of the time does your optimal agent beat the random agent?
    qs = Counter()
    cs = Counter()
    pi = {}

    updated = True

    while updated:
        updated = False
        h = get_trajectory(env, p1, p2)
        w = 1
        for i, (s, a, _) in enumerate(h):
            g = sum([r for (_, _, r) in h[i+1:]])
            cs[(s, a)] += w
            qs[(s, a)] += (w / cs[(s, a)])*(g - qs[(s, a)])
            max_a = None
            max_q = None
            for a in range(1, env.die_nsides + 1):
                q = qs[(s, a)]
                if (max_a is None and max_q is None) or max_q < q:
                    max_a = a
                    max_q = q                                    
            if s not in pi or pi[s] != max_a:
                pi[s] = max_a
                updated = True
            else:
                w *= 1 / get_action_probility_for_state((s[0], s[1], p1.token), a, p1)
    
    opt_agent = BasicDeterministicPolicy()
    opt_agent.get_action = lambda s : (pi[s], 1.0)
    return opt_agent


def play_game(player1, player2):
    # Demo of the agents playing the game
    game = MiniGammon5Env()
    numsteps = 0
    done = game.done()
    # uncomment the below to alias players with results from value iteration
    while not done:
        try:
            # print("\n===================================\n" + str(game) + "\n===================================\n")
            if player1 and game.turn == player1.token:
                action, _ = player1.get_action(game)
            elif player2 and game.turn == player2.token:
                action, _ = player2.get_action(game)
            else:
                # if game.out()[game.turn]: 
                #     print("First move must roll in (list roll)")
                # print("Move {} (format (roll, from, to))>".format(game.turn), end=' ')
                action = eval(input() or "None")
            game.step(action)
            numsteps += 1
        except MoveError as e:
            print('------------------------------------------------------------------------------------------')
            print("ERROR: " + str(e))
            print('------------------------------------------------------------------------------------------')
        done = game.done()

    # if done > 0:
    #     print('Player o won in {} moves!'.format(numsteps))
    # elif done < 0:
    #     print('Player x won in {} moves!'.format(numsteps))
    # else: assert False

    return done, numsteps


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
    parser.add_argument('--player1')
    parser.add_argument('--player2')

    args = parser.parse_args()

    game = MiniGammon5Env()
    print('Starting configuration')
    print(game)
    print(game.board, game.dice, game.turn)

    player1, player2 = None, None
    if args.player1:
        player1 = eval(args.player1)()
    if args.player2:
        player2 = eval(args.player2)() 
    print('player1 ({}) is {}'.format(player1.token, player1.__class__.__name__))
    print('player2 ({}) is {}'.format(player2.token, player2.__class__.__name__))

    states = enumerate_states(game.get_start_board(), set(), 1)
    print('----- {} total states -----'.format(len(states)))
    # Pick an arbitrary state to test
    test_state_x = ((-2, 0, 2, 0, 0), 1, 'x')
    test_state_o = None
    for state in list(states)[100:]:
        # pick an early state, but not the first state
        _, _, turn = state
        if test_state_x and test_state_o: break
        if turn == 'o': test_state_o = state
    print('Test states:', test_state_x, test_state_o)

    print('===== Testing get_trajectory =====')
    test_trajectory(game, player1, player2)
    
    
    print('====== Values (1) -- on-policy exploring starts =====')
    vs_on_es = on_policy_exploring_starts(game, player1, player2, states)
    assert player1.token == 'x'
    print('Value at arbitrary state {}: {}'.format(test_state_x, vs_on_es[test_state_x]))

    print('====== Values (2) -- on-policy soft ===== ')
    # When testing this function, you will want to update min_obs. Feel free
    # add this as an argument to parser, if you are exectuing this script in batch
    # from the terminal.
    vs_on_soft = on_policy_soft(game, player1, player2, min_obs=0)
    for s in vs_on_soft.keys():
        (board, roll) = s
        if (board, roll, player1.token) not in states:
            # enumerate_states skips this board (for both die rolls). Haven't had time to debug. 
            # Extra credit 1 point for fixing enumerate_states to detect it (without hard-coding of course! Only clever solutions :))
            #if board == (0, 2, 0, -2, 0): pass
            print(s)
    exit()
    try:
        print('Value at arbitrary state {}: {}'.format(test_state_x, vs_on_soft[test_state_x]))
    except KeyError:
        print('DID NOT VISIT STATE', test_state_x)

    print('===== Values (3) -- off-policy IS =====')
    vs_off_is = importance_sampling(game, player2, player1)
    assert player2.token == 'o'
    print('Value at arbitrary state {}: {}'.format(test_state_o, vs_off_es[test_state_o]))

    print('===== Off policy Monte Carlo Control =====')
    opt_agent = off_policy_mc_control(game, player1, player2)

    # # Have the optimal agent play against the random agent
    player2 = opt_agent
    print('Optimal player is {}'.format(player2.token))

    wins = {'x' : 0, 'o' : 0}
    for _ in range(100):
        winner, num_steps = play_game(player1, player2)
        wins[BackgammonEnv.get_token[winner]] += 1
    print('Total wins for 100 trials\n\tPlayer x wins: {}\n\tPlayer o wins: {}'.format(wins['x'], wins['o']))