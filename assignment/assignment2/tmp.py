import pickle
from assignment2 import *

with open("sida.pickle", "rb") as f:
    pi1, pi2, pi1_, pi2_, player1, player2, v1, v2, v1_vi, v2_vi = pickle.load(f)

# for i, value in v1_vi.items():
#     if v1[i]!=value:
#         print(f"? {v1[i]} {value}")
#     else:
#         print(f"=========> {v1[i]} {value}")

game = MiniGammon5Env()

states = enumerate_states(game)
# c = 0
# for state in states:
#     # if v1_vi[state]==0:
#     #     if np.sum(state)==0:
#     #         if np.sum(np.array(state)==2)==1 and np.sum(np.array(state)==-2)==1:
#     #             print(state)
#     if v1_vi[state]==-1.11:
#         c+=1
#         print(state)
#         p = player1
#         p2 = player2
#         game.set_state(state, [1], p.token)

#         for i in range(4):
#             a = p.get_action(game)
#             print(a)
#             if a is not None:
#                 game.step(a)
#                 print(game.board)
#                 a = p2.get_action(game)
#                 print(a)
#                 if a is not None:
#                     game.step(a)
#                     print(game.board)


#         print("="*10)
# print(f"Total {c}")


state = (1, 0, 0, -2, 1)
print(state)
p = BasicDeterministicPolicy()
p2 = RandomAgent()
game.set_state(state, [1], p.token)
for i in range(4):
    a = p.get_action(game)
    print(a)
    if a is not None:
        game.step(a)
        print(game.board)
        a = p2.get_action(game)
        print(a)
        if a is not None:
            game.step(a)
            print(game.board)