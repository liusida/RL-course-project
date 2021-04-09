import math
import numpy as np

from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Any, Union

np.random.seed(0)

#%% [markdown]
# # Programming "Assignemnt" 3
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

        # Sida:
        if to_pos<0 or from_pos>self.size-1:
            raise MoveError("BUG.")
        # End Sida.

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
 

class MiniGammon5Env(BackgammonEnv):


    def __init__(self):
        BackgammonEnv.__init__(self, size=5, ntokens=2, homesize=1, ndice=1, die_nsides=2)


    def get_start_board(self) -> Tuple[int]:
        board = self.size*[0]

        board[-1] =  2
        board[0]  = -2

        return board


    def reward(self, action, player_token):
        # Sida: This reward function is asking the agent to end the game as soon as possible, no matter who wins.
        player_sign = BackgammonEnv.get_sign[player_token]

        if player_sign == -1: 
            if self.board[-1] == -2:
                # This player has won. :)
                return 0
            elif self.board[0] == 2: 
                # The opposition player has won. :(
                return -2
                # return -2000 # Sida: otherwise how can an agent want to win the game?

        if player_sign == 1:
            if self.board[0] == 2: 
                # This player has won. :)
                return 0
            elif self.board[-1] == -2: 
                # The opposition player has won. :(
                return -2
                # return -2000 # Sida: otherwise how can an agent want to win the game?

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

# def enumerate_states(board, states, sign):
#     env = MiniGammon5Env()
#     token = BackgammonEnv.get_token[sign]

#     # Shouldn't have to check env.done, since env should return the same state when done

#     for roll in range(1, env.die_nsides + 1):
#         env.set_state(board, [roll], turn=token)
#         s = (tuple(board), roll, token) 
#         if s in states: continue
#         states.add(s)

#         # If the current player has tokens out, they need to roll in. 
#         # Since there is only one die, they don't have any options for rolling in. 
#         if env.out()[token]:
#             to_pos = len(board) - roll if sign > 0 else roll - 1
#             action = (roll, None, to_pos)
#             try:
#                 env.verify_action(*action, turn=token)
#                 env.step(action)
#                 enumerate_states(env.board, states, (sign * -1))
#             except MoveError: pass

#         else:
#             for i in Agent.get_possible_from_pos(env, token):
#                 env.set_state(board, [roll], turn=token)
#                 action = (roll, i, i + (roll * env.get_direction(token)))
#                 try:
#                     env.verify_action(*action, turn=token)
#                     env.step(action)
#                     enumerate_states(env.board, states, (sign * -1))
#                 except MoveError: pass
    
#     return states
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
    
    # if no_move:
    #     s = (tuple(board), None, token) 
    #     if s not in states:
    #         env.step(None)
    #         states.add(s)
    #         enumerate_states(env.board, states, (sign * -1))

    return states

def test_trajectory(env, p1, p2):
    for i in range(1000):
        env.reset()
        if i % 100 == 0: print('{}th iteration...'.format(i))
        t = get_trajectory(env, p1, p2)
        last_reward = t[-1][2]
        # we either won or we lost, or had a draw
        # just can't stop if there are more moves
        assert last_reward != -1, 'Reward: {}'.format(last_reward) 


def get_action_probility_for_state(s: State, a: Action, pi: Agent) -> float:
    # returns pi(a|s)
    # agents' get_action methods need an env argument
    env = MiniGammon5Env()
    count_retry = 0
    while True:
        board, roll, turn = s
        env.set_state(board, roll, turn=turn)
        actual, p = pi.get_action(env)
        if actual == a: return p
        count_retry += 1
        if count_retry>10:
            print("retry > 10.")


def play_game(player1, player2):
    # Demo of the agents playing the game
    game = MiniGammon5Env()
    numsteps = 0
    done = game.done()
    # uncomment the below to alias players with results from value iteration
    for _ in range(1000):
        if done:
            break
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


# %% [markdown]
# This is the actual assignment portion.

def debug(state):
    """ Print the state on the screen 
    the state is in the student's version, no [roll] but roll.
    """
    _board, _roll, _turn = state
    _env = MiniGammon5Env()
    _env.set_state(_board, [_roll], _turn)
    print(_env)

def play_until_ends(env: MiniGammon5Env, p1: Agent, p2: Agent):
    assert p1.token!=p2.token
    trajectory = []
    memo_history = []
    p1_rewards, p2_rewards = 0, 0
    numsteps = 0
    done = env.done()
    memo_history.append(env.get_state()) #
    while not done:
        # always start with p1
        assert env.turn == p1.token
        old_state = env.get_state()
        try:
            action_p1, _ = p1.get_action(env)
            env.step(action_p1)
            memo_history.append(env.get_state()) #
            if env.done():
                trajectory.append( (old_state, action_p1, env.reward(action_p1,p1.token)) )
                p1_rewards += env.reward(action_p1,p1.token)
                break
            action_p2, _ = p2.get_action(env)
            env.step(action_p2)
            memo_history.append(env.get_state()) #
        except MoveError as e:
            print('------------------------------------------------------------------------------------------')
            print("ERROR: " + str(e))
            print('------------------------------------------------------------------------------------------')

        numsteps += 1
        trajectory.append( (old_state, action_p1, env.reward(action_p1,p1.token)) )
        p1_rewards += env.reward(action_p1,p1.token)
        done = env.done()
    return trajectory, p1_rewards, memo_history

def get_trajectory(env: MiniGammon5Env, p1: Agent, p2: Agent) -> List[Tuple[State, Action, int]]:
    # (1 point) Get a sample trajectory. 
    #
    # Hints:
    # * Make sure you include the final reward (remember the indexing of trajectories).
    # * When stepping through a time unit, make sure you check whether the first player has 
    #   has won before having the second player take their turn.
    assert p1.token!=p2.token
    ret = []
    env.reset()
    ret, _, memo_history = play_until_ends(env, p1, p2)
    return ret

def on_policy_exploring_starts(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent, states: List[Tuple[Board, int, str]]) -> Dict[State, float]:
    # (1 point) Compute the value function of the RandomAgent with exploring starts
    # Compute the average returns for 30 sample trajectories.
    # In the reflection, report the value of the arbitrarily chosen state ((1, 1, 0, 0, -1), 1, 'x'), for the RandomAgent.
    #
    # Hints:
    # * You may be tempted to re-use trajectories; you do not need to. Feel free to do independent draws for each state.
    
    # This is a naive implementation that doesn't leverage the fact that some states appear in the trajectories we will be generating.
    """ The algorithm on Page 92 """
    ret = {}
    rewards = defaultdict(list)
    for i in range(30):
        print(f"iteration {i}/30")
        for state in states:
            board, dice, turn = state
            if turn=='x': # only consider player1
                env.set_state(board, [dice], turn)
                trajectory, reward, memo_history = play_until_ends(env, p1, p2)
                if len(trajectory)==0:
                    continue # There are many states that one already wins.
                _reward = 0
                for t in range(len(trajectory)-1, -1, -1):
                    _t = trajectory[t]
                    _state = _t[0]
                    _action = _t[1]
                    _reward += _t[2]
                    rewards[_state].append(_reward)
                assert _reward == reward
    for state in states:
        state = formatted_state(state)
        if state[2]=='x': # only consider player1
            ret[state] = np.mean(rewards[state])
    return ret 


class OptimizibleRandomAgent(RandomAgent):
    """ A wrapper for on-policy soft RandomAgent """
    def __init__(self, token='x'):
        self.policy = {}
        self.epsilon = 0.1
        super().__init__(token)
    def get_action(self, env) -> Tuple[Action, float]:
        state = env.get_state()
        if state in self.policy:
            epsilon = np.random.random()
            if epsilon>self.epsilon:
                return self.policy[state], 1.0
            # else is just like the random agent
        return super().get_action(env)
    def set_action(self, state, action):
        """ set actions while optimizing """
        self.policy[state] = action

def on_policy_soft(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent, min_obs=0, states=None):
    """ How could we make a on-policy algorithm for a RandomAgent?? Will a RandomAgent follow any action function other than random?"""
    # (1 point) Compute the value function of the RandomAgent by allowing it to explore. You may choose to use 
    # the enumerated states to initialize your value function, but this is not required.
    # 
    # In the reflection, report the value of the arbitrarily chosen state ((1, 1, 0, 0, -1), 1, 'x'), for 
    # the RandomAgent, and answer the questions: How many sample trajectories did you take in order to converge?
    # What is the smallest number of observations you encountered for your set of state-action pairs?
    #
    # Set a minimum number of observations. What number did you choose? Why?
    #
    # Now update the function to only return if it has seen at least this number of state-action pairs *and*
    # has converged. How many sample trajectories did you draw? Given infinite samples, soft policies should 
    # take every possible action for each state they encounter. Did your agent fail to encoutner any states?
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
    """ The algorithm on Page ?? """
    player1 = OptimizibleRandomAgent()

    num_states = len(g_states)
    q_value = crazy_nested_dictionary() # [(state, action)] -> state-action-value
    # value_function = {} # [state] -> state-value
    rewards = defaultdict(list) # [(state, action)] -> a reward list
    for i in range(30):
        print(f"iteration {i}/30")
        for state in g_states:
            board, dice, turn = state
            if turn=='x': # only consider player1
                env.set_state(board, [dice], turn)
                trajectory, reward, memo_history = play_until_ends(env, player1, p2)
                _reward = 0
                if len(trajectory)==0:
                    continue
                for t in range(len(trajectory)-1, -1, -1):
                    _t = trajectory[t]
                    _state = _t[0]
                    _action = _t[1]
                    _reward += _t[2]
                    rewards[(_state, _action)].append(_reward)
                    number_obs = len(rewards[(_state, _action)])
                    if number_obs>min_obs: # set action only when visit the same state and action more than min_obs times.
                        q_value[_state][_action] = np.mean(rewards[(_state, _action)])
                        argmax_action = list(q_value[_state].keys())[np.argmax(q_value[_state].values())]
                        # print(argmax_action)
                        player1.set_action(state=_state, action=argmax_action)
    return player1.policy

def formatted_state(state):
    _a,_b,_c = state
    formatted_state = _a,(_b,),_c
    return formatted_state

def importance_sampling(env: BackgammonEnv, pi: SoftDeterministicAgent, b: RandomAgent, min_obs=0, states=None):
    # (1 point) Implement importance sampling for estimating value functions. You may choose to use 
    # the enumerated states to initialize your value function, but this is not required.
    #
    # In the reflection, report the value of the arbitrarily chosen state ((1, 1, 0, 0, -1), 1, 'x'), for 
    # the RandomAgent, and answer the questions: How many sample trajectories did you take in order to converge?
    # What is the smallest number of observations you encountered for your set of state-action pairs?
    #
    # Set a minimum number of observations. What number did you choose? Why?
    #
    # Now update the function to only return if it has seen at least this number of state-action pairs *and*
    # has converged. How many sample trajectories did you draw? Given infinite samples, soft policies should 
    # take every possible action for each state they encounter. Did your agent fail to encoutner any states?

    # Hints:
    # * Your implementation can more or less follow the algorithm in the book/lectures
    # * You will want to play around with your convergence condition, especially if you 
    #   choose not to use a discount factor. I recommend a multiplicative convergence 
    #   condition, similarly to the basic monte carlo sampling you implemented for 
    #   soft policies.

    """
    Warning: this is to compute value function for player 'o'!

    Assuming b is the behavior policy, pi is the target_policy.
    Let's create an opponent.
    """
    behavior_policy = b
    behavior_policy.token = 'o'
    target_policy = pi
    target_policy.token = 'o'
    opponent = SoftDeterministicAgent(token='x')

    def importance_sampling_ratio(target_policy,behavior_policy,action_state_pairs):
        assert target_policy.token==behavior_policy.token
        numerator, denominator = 1,1
        _target_policy = BasicDeterministicPolicy(token='x')
        for action, state in action_state_pairs:
            _env = MiniGammon5Env()
            _env.set_state(*state)

            _action, _ = _target_policy.get_action(_env)
            if _action==action:
                _p = 0.9
            else:
                _p = 0.1
            numerator *= _p

            _, _p = b.get_action(_env)
            denominator *= _p

        return numerator/denominator

    num_states = len(g_states)
    value_function = {} # [state] -> state-value
    rewards = defaultdict(list)
    importance_ratios = defaultdict(list)
    rewards_times_importance_ratios = defaultdict(list)
    how_many_trajectories = 0
    while True:
        for state in g_states:
            board, dice, turn = state
            if turn==behavior_policy.token:
                env.set_state(board, [dice], turn)
                trajectory, reward, memo_history = play_until_ends(env, behavior_policy, opponent) # b play against another_player, not pi
                how_many_trajectories += 1
                # for t in trajectory:
                #     debug(t[0])
                #     print(t[1], t[2])

                reward = 0
                if len(trajectory)==0:
                    continue
                    value_function[state] = reward
                action_state_pairs = []
                for j in range(len(trajectory)-1, -1, -1):
                    step_j = trajectory[j]
                    reward += step_j[2]
                    action = step_j[1]
                    _state = step_j[0]
                    assert isinstance(reward, int)
                    if _state[2]==b.token: # only add valid state
                        action_state_pairs.append((action,_state))
                        ratio = importance_sampling_ratio(target_policy,behavior_policy,action_state_pairs)

                        rewards[_state].append(reward)
                        importance_ratios[_state].append(ratio)
                        rewards_times_importance_ratios[_state].append(ratio*reward)
        


        numerator, denominator = [],[]
        for state in g_states:
            if state[2]==behavior_policy.token:
                state = formatted_state(state)
                numerator = np.sum(rewards_times_importance_ratios[state])
                denominator = np.sum(importance_ratios[state])
                if numerator==denominator==0.0:
                    value_function[state] = 0.0
                else:
                    value_function[state] = numerator / denominator # This is weighted value-function
        totally_done = True
        for state in g_states:
            if state[2]==behavior_policy.token:
                state = formatted_state(state)
                if state not in value_function.keys():
                    totally_done = False
        if totally_done:
            break
        print(f"len {len(value_function)} v.s. {num_states}")
    print(f"How many trajectories: {how_many_trajectories}")
    return value_function

class OffPolicyMcAgent(Agent):
    def __init__(self, token='x'):
        self.token = token
        self.policy = {}
        self.backup = RandomAgent(token=token)

    def set_action(self, state, action):
        self.policy[state] = action

    def get_action(self, env: BackgammonEnv) -> Tuple[Action, float]:
        state = env.get_state()
        if state in self.policy:
            return self.policy[state], 1.0
        else:
            print(f"State {state} is not in the policy, use a backup plan.")
            return self.backup.get_action(env)

def prob_of_random_action(state):
    _b = RandomAgent()
    _env = MiniGammon5Env()
    _env.set_state(*state)
    _, _p = _b.get_action(_env) # with equal probability
    return _p

def crazy_nested_dictionary():
    return defaultdict(crazy_nested_dictionary)

def off_policy_mc_control(env: BackgammonEnv, p1: RandomAgent, p2: SoftDeterministicAgent):
    # (1 point) Implement off policy monte carlo control, to produce an agent. 
    #
    # Monte Carlo methods are not guaranteed to produce an optimal agent in finite steps. 
    # What percentage of the time does your optimal agent beat the random agent?
    assert p1.token!=p2.token
    learning_agent = OffPolicyMcAgent(p1.token)
    cumulative_sum_of_weights = crazy_nested_dictionary()
    q_value = crazy_nested_dictionary()

    for i in range(5):
        for _ in range(2):
            print(". ", end="")
            for state in g_states:
                board, dice, turn = state
                env.set_state(board, [dice], turn)
                if turn!=learning_agent.token:
                    continue
                trajectory, _, memo_history = play_until_ends(env, p1, p2)
                if len(trajectory)==0:
                    continue
                rewards = 0
                weight = 1
                for j in range(len(trajectory)-1, -1, -1):
                    _step_t = trajectory[j]
                    _state_t = _step_t[0]
                    if _state_t[2]!=learning_agent.token:
                        continue
                    _action_t = _step_t[1]
                    _reward_t = _step_t[2]
                    rewards += _reward_t # discount 1.0
                    if type(cumulative_sum_of_weights[_state_t][_action_t])==defaultdict:
                        cumulative_sum_of_weights[_state_t][_action_t] = 0
                    cumulative_sum_of_weights[_state_t][_action_t] += weight
                    if type(q_value[_state_t][_action_t])==defaultdict:
                        q_value[_state_t][_action_t] = 0
                    q_value[_state_t][_action_t] += weight/cumulative_sum_of_weights[_state_t][_action_t] * (rewards-q_value[_state_t][_action_t])
                    argmax_action = list(q_value[_state_t].keys())[np.argmax(q_value[_state_t].values())]
                    learning_agent.set_action(_state_t, argmax_action)
                    if _action_t!=learning_agent.policy[_state_t]:
                        break
                    weight /= prob_of_random_action(_state_t)


        print("Test")

        # qs = []
        # for i1,q1 in q_value.items():
        #     for i2,q2 in q1.items():
        #         qs.append(q2)
        #         if q2>-200:
        #             print(i1, i2, q2)
        # print(np.mean(qs))
        # import matplotlib.pyplot as plt
        # plt.hist(qs, bins="auto")
        # plt.show()
        

        # test_random_player = RandomAgent(token='o')
        # wins = []
        # for i in range(1000):
        #     env.reset()
        #     trajectory, _, memo_history = play_until_ends(env, learning_agent, test_random_player)
        #     done = env.done() # -1 is winning
        #     # if done!=-1:
        #     #     debug_show_history(trajectory, memo_history, learning_agent, q_value)
        #     wins.append(done==-1)
        # print(f"learning agent wins {100*np.mean(wins)}%.")
        
        # Update the behave policy
        # p1 = learning_agent
    return learning_agent

def debug_show_history(trajectory, memo_history, learning_agent, q_value):
    for i in range(len(memo_history)):
        if i%2==0:
            j = int(i/2)
            if j<len(trajectory):
                state = trajectory[j][0]
                print(i, "q_value>", dict(q_value[state]))
                print(i, "action>", trajectory[j])
        else:
            print(i, "board>  ", memo_history[i])
    exit(0)
    return

def debug_show_history_simple(trajectory, memo_history):
    for i in range(len(memo_history)):
        print("")
        if i%2==0:
            j = int(i/2)
            if j<len(trajectory):
                state = trajectory[j][0]
                print(i, "action>", trajectory[j])
            else:
                print(i, "final> ", memo_history[i])
        else:
            print(i, "board>  ", memo_history[i])
    return

def output_sample_trajectory():
    game = MiniGammon5Env()
    player1 = RandomAgent()
    player2 = SoftDeterministicAgent()
    trajectory, _, memo_history = play_until_ends(game, player1, player2)
    debug_show_history_simple(trajectory, memo_history)

def reproduce_bug():
    env = MiniGammon5Env()
    env.set_state((-2,1,1,0,0), [2], 'o')
    p2 = SoftDeterministicAgent()
    action, _ = p2.get_action(env)
    print(action, env.board)
    env.step(action)
    print(env.board)

# %%
if __name__ == "__main__":
    from itertools import chain
    env = MiniGammon5Env()
    states_1 = enumerate_states(env.get_start_board(), set(), -1)
    print(len(states_1), sorted(list(states_1))[0])
    states_2 = enumerate_states(env.get_start_board(), set(), 1)
    print(len(states_2), sorted(list(states_2))[0])

    # use g_states instead
    g_states = sorted(chain(states_1,states_2))
    print(len(g_states), g_states[150])
        

    output_sample_trajectory()

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
    parser.add_argument('--player1', default="RandomAgent")
    parser.add_argument('--player2', default="SoftDeterministicAgent")

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

    states = g_states
    print('----- {} total states -----'.format(len(states)))
    # Pick an arbitrary state to test
    test_state_x = ((-2, 0, 2, 0, 0), (1,) , 'x')
    test_state_o = None
    for state in list(states)[100:]:
        # pick an early state, but not the first state
        _, _, turn = state
        if test_state_x and test_state_o: break
        if turn == 'o': test_state_o = formatted_state(state)
    print('Test states:', test_state_x, test_state_o)

    print('===== Testing get_trajectory =====')
    test_trajectory(game, player1, player2)
    
    test_state_x = ((1, 1, 0, 0, -1), (1,), 'x')
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
        (board, roll, _token) = s
        if (board, roll[0], player1.token) not in states:
            # enumerate_states skips this board (for both die rolls). Haven't had time to debug. 
            # Extra credit 1 point for fixing enumerate_states to detect it (without hard-coding of course! Only clever solutions :))
            if board == (0, 2, 0, -2, 0): pass
            print(s)
    try:
        print('Value at arbitrary state {}: {}'.format(test_state_x, vs_on_soft[test_state_x]))
    except KeyError:
        print('DID NOT VISIT STATE', test_state_x)

    print('===== Values (3) -- off-policy IS =====')
    vs_off_is = importance_sampling(game, player2, player1)
    assert player2.token == 'o'
    print('Value at arbitrary state {}: {}'.format(test_state_o, vs_off_is[test_state_o]))

    print('===== Off policy Monte Carlo Control =====')
    player1.token, player2.token = 'x','o'
    opt_agent = off_policy_mc_control(game, player1, player2)

    # # Have the optimal agent play against the random agent
    player1 = opt_agent
    player2 = RandomAgent(token='o')
    print('Optimal player is {}'.format(player1.token))

    wins = {'x' : 0, 'o' : 0}
    for _ in range(100):
        winner, num_steps = play_game(player1, player2)
        if winner!=0:
            wins[BackgammonEnv.get_token[winner]] += 1
    print('Total wins for 100 trials\n\tPlayer x wins: {}\n\tPlayer o wins: {}\n\tDraw: {}'.format(wins['x'], wins['o'], 100-wins['x']-wins['o']))

# Format
# action -> (roll, from, to)