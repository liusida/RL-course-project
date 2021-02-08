import math
import typing
import numpy as np
np.random.seed(1234)

#%% [markdown]
# # Programming Asignment 1
# The goal of this first programming assignment is to give you the opportunity to build some intuitions about the RL framework. We will not be programming any RL algorithms just yet. Instead, you still start by hand-coding an agent to play a constrained and deterministic version of Backgammon.
# 
#
# # Backgammon
# Backgammon is an ancient game, but it also has historical significance within RL. There are many features that make a suitable environment for the general RL framework, including:
# 
# * Delayed rewards -- the optimal play may involve sacrificing a token and seemingly falling behind in the next few time steps, but that move can also substantially increase the expected rewards.
# * Stochasticity via dice rolls (we will start by fixing the random seed)
# * Multiple agents (we will start by encoding a deterministic adversary that we treat as embedded in the environment)
# * Stages of gameplay that could benefit from different strategies (i.e., there may be subtasks or important contextual information)
# * The ability to define a contrained version of the problem that has an analytical solution
# * Some actions only permissible in certain contexts (e.g., skip when there are no valid moves -- we are simplifying to allow skip for every time step)
# * Turns correspond to multiple steps in the environment

#%% [markdown]
# # The Environment
# Below is the environment you will use for your agents. You can test out how the game works an interact with it by executing this file. 

class MoveError(Exception): pass

class BackgammonEnv():

    get_sign = {'x' : -1, 'o': 1}    


    def __init__(self, other):
        self.turn = 'x'
        self.dice = None
        self.board = None
        self.out = None
        self.reset()


    def reset(self):
        self.out = {'x': 0, 'o': 0}

        self.board = 24*[0]

        self.board[-1] = 2
        self.board[12] = 5
        self.board[7]  = 3
        self.board[5]  = 5

        self.board[0]   = -2
        self.board[-13] = -5
        self.board[-8]  = -3
        self.board[-6]  = -5

        self.roll()


    def done(self):
        x_home = self.board[-6:] 
        x_sign = BackgammonEnv.get_sign['x']
        num_x_in_home = sum([abs(x) for x in x_home if math.copysign(-1, x) == x_sign])

        o_home = self.board[:6]
        o_sign = BackgammonEnv.get_sign['o']
        num_o_in_home = sum([abs(o) for o in o_home if math.copysign(-1, o) == o_sign])

        if num_x_in_home == 15: return x_sign
        if num_o_in_home == 15: return o_sign
        return 0


    def roll(self): 
        self.dice = list(np.random.choice(6, 2) + [1,1])
        if self.dice[0] == self.dice[1]:
            self.dice = [*self.dice, *self.dice]


    def same_sign_at(self, pos):
        if self.board[pos]:
            return math.copysign(1, BackgammonEnv.get_sign[self.turn]) == math.copysign(1, self.board[pos])
        else: return True


    def verify_action(self, roll, from_pos, to_pos): 
        # while the player has tokens off the board, they must try to roll in
        if self.out[self.turn]:
            if self.turn == 'x':
                if to_pos != roll - 1:
                    raise MoveError("Must roll in before moving!")
            else:
                if to_pos != 24 - roll:
                    raise MoveError("Must roll in before moving!")
            # cannot roll in on a position where the opponent has > 1 token
            if not self.same_sign_at(to_pos) and abs(self.board[to_pos]) > 1:
                raise MoveError("Cannot roll in on {}".format(to_pos))
            return
        
        # player must select a valid position on the board
        if from_pos < 0: 
            raise MoveError("Position {} is not on the board".format(from_pos))
        if to_pos > 23:
            raise MoveError("Position {} is not on the board".format(to_pos))

        # player must have a token at the start position
        if not self.same_sign_at(from_pos):
            raise MoveError("Player {} has no tokens at position {}".format(self.turn, from_pos))

        # player can only move to locations with at most one enemy token
        if not self.same_sign_at(to_pos):
            if abs(self.board[to_pos]) > 1:
                raise MoveError("Cannot move {} tokens to Player {}'s position ({})".format(self.turn, 'x' if self.turn == 'o' else 'o', to_pos))
        
        # x/self moves clockwise/increasing
        # make sure the moves add up
        if self.turn == 'x' and (from_pos + roll) != to_pos:
            raise MoveError("Player {} must move clockwise/increasing and from_pos + roll == to_pos".format(self.turn))
        elif self.turn == 'other' and (from_pos - roll) != to_pos:
            raise MoveError("Player {} must move counter-clockwise/decreasing and from_pos - roll == to_pos".format(self.turn))


    def apply_action(self, roll, from_pos, to_pos):
        sign = BackgammonEnv.get_sign[self.turn]
        
        if not self.same_sign_at(to_pos):
            self.out['x' if self.turn == 'o' else 'o'] += 1
            self.board[to_pos] = sign
        else: 
            self.board[to_pos] += sign

        if self.out[self.turn]:
            self.out[self.turn] -= 1
        else:
            self.board[from_pos] -= sign


    def verify_roll(self, roll):
        if roll not in self.dice:
            raise MoveError("{} not in {}".format(roll, self.dice))


    def next_turn(self):
        self.turn = 'x' if self.turn == 'o' else 'o'
        self.roll()


    def step(self, action):
        if action is None or action == 'None': 
            self.next_turn()
            return

        if type(action) is int:
            action = action, None, action - 1 if self.turn == 'x' else 24 - 1

        roll = action[0]

        self.verify_roll(roll)      
        self.verify_action(*action)
        self.apply_action(*action)

        self.dice.remove(roll)
        
        if not self.dice: self.next_turn()
        

    def __str__(self):
        retval = "Player: {}, Roll: {}\nOut: {}\n\n".format(self.turn, self.dice, self.out[self.turn])
        get_line = lambda cells: "  ".join(["o" if c > 0 else "x" if c < 0 else " " for c in cells])

        selfs_home = self.board[:12]
        oppos_home = self.board[12:]
        selfs_home.reverse()
        
        selfs_indices = [str(i).zfill(2) for i in range(0, 12)]
        selfs_indices.reverse()
        oppos_indices = range(12, 24)

        retval += " ".join([str(i) for i in oppos_indices]) + "\n"
        
        while True:
            retval += get_line(oppos_home) + "\n"
            if sum(abs(np.array(oppos_home))) == 0: break                
            else: 
                for i, c in enumerate(oppos_home):
                    if c > 0: oppos_home[i] -= 1
                    elif c < 0: oppos_home[i] += 1

        lines_to_rev = []
        while True:
            lines_to_rev.append(get_line(selfs_home))
            if sum(abs(np.array(selfs_home))) == 0: break
            else:
                for i, c in enumerate(selfs_home):
                    if c > 0: selfs_home[i] -= 1
                    elif c < 0: selfs_home[i] += 1

        lines_to_rev.reverse()
        return retval + "\n".join(lines_to_rev) +  "\n" + " ".join(selfs_indices)


# %% [markdown]
# ## Program two agents (4 points)
# You will code up two agents below. To receive full credit for each, make sure you fill out the associated questions on Blackboard.

# %%
def RandomAgent(): 
    # 1 point: Code a "stochastic" agent and describe your design decisions (e.g., what space are you sampling over?)
    # * This agent should use functions in the np.random namespace, so that it starts out as _deterministic_
    # * Write your description of your design decisions in the associated Blackboard question.
    pass                

# %%
def get_observation(game):
    # 1 point: Code a representation of the current state in the environment that your eventual RL agent will observe and describe your design decisions.
    # * What features of the environment will you use directly?
    # * Will there be any constructed or computed features?
    # * White your description of your design decisions in the associated Blackboard questions.

    default_player_symbol = game.turn
    player_symbols = [ default_player_symbol, 'x' if default_player_symbol=='o' else 'o']

    board = np.zeros(shape=[24,15,2])
    out = np.zeros(shape=[1,15,2])
    turn = np.zeros(shape=[1])
    turn[0] = (game.turn == default_player_symbol)
    print(f"=== {default_player_symbol} ===")
    for player in range(2):
        print(f"player {'self' if player==0 else 'opponent'}")
        piece = 0
        for point, value in enumerate(game.board):
            if math.copysign(-1, value) != game.get_sign[player_symbols[player]]:
                continue
            if value<0:
                value = -value
            for i in range(value):
                # rearrange the order of obs, so it will be similar to the observation of player 1, with 2 pieces in the starting point.
                _point = point
                _piece = piece
                if player_symbols[player]!='x':
                    _point = 23-point
                    _piece = 14-piece
                board[_point, _piece, player] = 1
                piece += 1
        print("")
        print(board[:,:,player])

        if player==0:
            sym = default_player_symbol
            player_symbols[player]
        else:
            sym = 'x' if default_player_symbol=='o' else 'o'
            player_symbols[player]
        for i in range(game.out[sym]):
            out[0,piece,player] = 1
            piece += 1
        assert piece==15

    # observation is the combination of these three components:
    obs = [board.flatten(), out.flatten(), turn.flatten()]
    obs = np.concatenate(obs)
    return obs

# %%
def reward(state, action):
    # 1 point: Code a reward function for your `get_observation` function and describe your design decisions.
    # * What features of the state does the reward use?
    # * Does your reward function use any features of the action space?
    # * What about this reward function do you think will be useful for the agent during training?
    pass

#%%
def GreedyAgent():
    # 1 point: Code an agent that uses the reward function to choose between actions. 
    # * How will you break ties between actions, if you encounter them?
    # * If you update your reward function after experimenting with the greedy agent, what did you change about your design choices?
    pass 


#%% [markdown]
# ## Evaluate your agents (1 point)

#%% 
# Write your code here. You will first need to write a driver loop that manages turns.
# Once your driver loop is completed, make sure you:
# 1. Run each agent until one wins (i.e., episode completion), and 
# 2. Track how much time it took for an episode to complete.
#
# Once you have run your agents, answer the following questions in Blackboard to recieve full credit:
# * Which agent won? Did that surprise you? Why or why not?
# * How long did it take for an episode to complete. Did you observe any unexpected outcomes?

def driver(agent1, agent2, game): pass

def evaluate(): pass

# %%
# If you'd like to 
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

    args = parser.parse_args()

    game = BackgammonEnv(None)

    if args.move_file:
        with open(args.move_file) as f:
            for move in f.readlines():
                print("\n===================================\n" + str(game) + "\n===================================\n")
                print(f"{move}")
                obs = get_observation(game)
                game.step(eval(move))


    while not (done := game.done()): 
        try:
            print("\n===================================\n" + str(game) + "\n===================================\n")
            if game.out[game.turn]: 
                print("First move must roll in (list roll)")
            print("Move {} (format (roll, from, to))>".format(game.turn), end=' ')
            obs = get_observation(game)
            action = eval(input() or "None")
            game.step(action)
        except MoveError as e:
            print('------------------------------------------------------------------------------------------')
            print("ERROR: " + str(e))
            print('------------------------------------------------------------------------------------------')

    if done > 0:
        print('Player o won!')
    elif done < 0:
        print('Player x won!')
    else: assert False
