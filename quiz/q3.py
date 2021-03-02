v = [0]*3
v_new = [0]*3

r = [1,-1,1]
ga = 0.5

for i in range(3000):
    for j in range(3):
        v_new[j] = r[j] + ga * v[(j+1)%3]
        print(f"{i}, {j}, {v_new[j]}")
    for j in range(3):
        v[j] = v_new[j]