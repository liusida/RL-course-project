import numpy as np

H1 = {
    "s": [3,10,46,100],
    "a": [1,2,1],
    "r": [-1,-1,-5],
}
H2 = {
    "s": [3,10,22,100],
    "a": [1,1,1],
    "r": [-1,-1,-1],
}
H3 = {
    "s": [3,9,41,100],
    "a": [1,3,2],
    "r": [-1,-10,-5],
}

q1 = 0
for i in range(1,3):
    q1 += H1["r"][i]
print("q1", q1)

q2 = 0
for i in range(0,3):
    q2 += H2["r"][i]
print("q2", q2)

def q3_pi(a, s): # probability of pi(a|s)
    prob = {
        1: 0.7 + 0.1, 
        2: 0.1,
        3: 0.1
    }
    return prob[a]
def q3_b(a, s): # probability of b(a|s)
    if s%2==0: #even
        # print("even state")
        prob = {
            1: 0.3,
            2: 0.7,
            3: 0.3,
        }
    else:
        # print("odd state")
        prob = {
            1: 0.8,
            2: 0.2,
            3: 0.8,
        }
    # assert sum(prob.values())==1.0
    return prob[a]
def q3_importance(h): # (5.3)
    importance = 1
    for i in range(3):
        tmp = q3_pi(h["a"][i], h["s"][i])
        importance *= tmp
        tmp = q3_b(h["a"][i], h["s"][i])
        importance /= tmp
    return importance
q3 = q3_importance(H2)
print("q3", q3)

q4_WG = 0
q4_G = 0
for h in [H1, H2, H3]:
    q4_WG += q3_importance(h)*sum(h["r"])
    q4_G += q3_importance(h)
q4 = q4_WG/q4_G # (5.6)
print("q4", q4)

print("#============================================")
