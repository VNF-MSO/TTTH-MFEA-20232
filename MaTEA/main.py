import numpy as np
import math
import random
import matplotlib.pyplot as plt

NP = 100
ACS = 150
reward_rate = 0.8
p = 0.8
knowledge_transfer_rate = 0.2
UR = 0.2
D = 50


class Cost():
    def __init__(self):
        self.cost_function = {0 : self.sphere, 1 : self.weierstrass, 2 : self.rosenbrock, 3 : self.griewank, 4 : self.rastrgin}

    def sphere(self, p):
        data = decode(p, -100, 100)
        f = []
        for z in data:
            v = 0
            for i in z:
                v += i * i
            f.append(v)
        return np.array(f)

    def weierstrass(self, p):
        a = 0.5
        b = 3
        k = 20
        data = decode(p, -0.5, 0.5)
        f = []
        for z in data:
            v = 0
            for i in range(25):
                for h in range(k + 1):
                    v += a ** h * math.cos(2 * math.pi * b ** k * (z[i] + 0.5))
            for h in range(k + 1):
                v -= a ** k * math.cos(2 * math.pi * b ** k * 0.5) * 25
            f.append(v)
        return np.array(f)

    def rosenbrock(self, p):
        data = decode(p, -50, 50)
        f = []
        for z in data:
            v = 0
            for i in range(49):
                v += (100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2)
            f.append(v)
        return np.array(f)

    def griewank(self, p):
        data = decode(p, -100, 100)
        f = []
        for z in data:
            v = 1
            s = 0
            m = 1
            for i in range(50):
                s += z[i] ** 2
                m *= math.cos(z[i] / math.sqrt(i + 1))
            v += (s / 4000 - m)
            f.append(v)
        return np.array(f)

    def rastrgin(self, p):
        data = decode(p, -50, 50)
        f = []
        for z in data:
            v = 0
            for i in z:
                v += (i ** 2 - 10 * math.cos(2 * math.pi * i) + 10)
            f.append(v)
        return np.array(f)
def init():
    sphere_p = np.random.rand(NP, D)
    sphere_ac =  sphere_p.copy()
    weierstrass_p = np.random.rand(NP, D)
    weierstrass_ac = weierstrass_p.copy()
    rosenbrock_p = np.random.rand(NP, D)
    rosenbrock_ac = rosenbrock_p.copy()
    griewank_p = np.random.rand(NP, D)
    griewank_ac = griewank_p.copy()
    rastrigin_p = np.random.rand(NP, D)
    rastrigin_ac = rastrigin_p.copy()

    list_p = {0 : sphere_p, 1 : weierstrass_p, 2 : rosenbrock_p, 3 : griewank_p, 4 : rastrigin_p}
    list_ac = {0: sphere_ac, 1: weierstrass_ac, 2: rosenbrock_ac, 3: griewank_ac, 4: rastrigin_ac}
    return list_p, list_ac, np.ones((5, 5)), np.ones((5, 5))

def sim(ac1, ac2):
    cov1 = np.cov(ac1, rowvar=False)
    cov2 = np.cov(ac2, rowvar=False)
    m1 = np.mean(ac1, axis=0)
    m2 = np.mean(ac2, axis=0)
    k1 = np.trace(np.linalg.inv(cov1) * cov2) + np.matmul(np.matmul((m2 - m1).T, np.linalg.inv(cov1)), (m2 - m1)) - cov1.shape[0] + np.log(abs(np.linalg.det(cov2)/np.linalg.det(cov1)))
    k1 = abs(k1)/2
    k2 = np.trace(np.linalg.inv(cov2) * cov1) + np.matmul(np.matmul((m2 - m1).T, np.linalg.inv(cov1)), (m2 - m1)) - cov2.shape[0] + np.log(
        abs(np.linalg.det(cov1) / np.linalg.det(cov2)))
    k2 = abs(k2) / 2
    return (k1+k2)/2

def choose_task(t, list_ac, r, s):
    f = 0
    task = []
    for i in range(5):
        if i != t:
            task.append(i)
            s[t][i] = p * s[t][i] + r[t][i]/np.log(sim(list_ac[t], list_ac[i]))
            f += s[t][i]
    proba = []
    for i in range(5):
        if i != t:
            proba.append(s[t][i]/f)
    q = []
    for i in range(4):
        v = 0
        for j in range(i+1):
            v += proba[j]
        q.append(v)
    r = random.random()
    if r < q[0]:
        return task[0]
    else:
        for i in range(3):
            if r >= q[i] and r < q[i+1]:
                return task[i + 1]

def knowledge_transfer(t, list_ac, list_p, r, s):
    assist = choose_task(t, list_ac, r, s)
    offsprings = []
    for i in range(NP):
        rnd_inv = random.randint(0, NP-1)
        k = random.randint(0, D - 1)
        v = []
        CRKTC = random.uniform(0.1, 0.9)
        for j in range(D):
            if j == k:
                v.append(list_p[assist][rnd_inv][j])
            else:
                rnd = random.random()
                if rnd < CRKTC:
                    v.append(list_p[assist][rnd_inv][j])
                else:
                    v.append(list_p[t][i][j])
        v = np.array(v)
        offsprings.append(v)
    return np.array(offsprings), assist

def DE(p):
    offsprings = []
    for i in range(NP):
        F = random.uniform(0.1, 2)
        r = random.randint(0, NP-1)
        w = p[i] + F*(p[r] - p[i])
        v = []
        k = random.randint(0, D-1)
        CR = random.uniform(0.1, 0.9)
        for j in range(D):
            if j == k:
                v.append(w[k])
            else:
                rnd = random.random()
                if rnd < CR:
                    v.append(w[j])
                else:
                    v.append(p[i][j])
        v = np.array(v)
        offsprings.append(v)
    return np.array(offsprings)

def generate(t, list_ac, list_p, r, s):
    rnd = random.random()
    if rnd > knowledge_transfer_rate:
        return DE(list_p[t]), t
    else:
        return knowledge_transfer(t, list_ac, list_p, r, s)

def update_archive(ac, p):
    for i in p:
        rnd = random.random()
        if rnd < UR:
            if ac.shape[0] < ACS:
                ac = np.append(ac, [i], axis=0)
            else:
                rnd_idx = np.random.randint(ac.shape[0])
                ac = np.delete(ac, rnd_idx, axis=0)
                ac = np.append(ac, [i], axis=0)

def update_subpop(p, offsprings, i, assist, r):
    cost = Cost()
    result_p = cost.cost_function[i](p)
    result_offsprings = cost.cost_function[i](offsprings)
    min_p = np.min(result_p)
    min_offsprings = np.min(result_offsprings)
    if i != assist:
        if min_offsprings < min_p:
            r[i][assist] = r[i][assist] / reward_rate
        else:
            r[i][assist] = r[i][assist] * reward_rate
    for j in range(NP):
        if result_offsprings[j] < result_p[j]:
            p[j] = offsprings[j]
    return np.min(cost.cost_function[i](p))

def main():
    list_p, list_ac, r, s = init()
    generation = 1000
    offsprings = []
    assists = []
    min = {0 : [], 1 : [], 2 : [], 3 : [], 4 : []}
    count_assist = []
    count_assist.append([0, 0, 0, 0, 0])
    count_assist.append([0, 0, 0, 0, 0])
    count_assist.append([0, 0, 0, 0, 0])
    count_assist.append([0, 0, 0, 0, 0])
    count_assist.append([0, 0, 0, 0, 0])
    for g in range(generation):
        print(f"Generation {g}:")
        for i in range(5):
            offspring, assist = generate(i, list_ac, list_p, r, s)
            offsprings.append(offspring)
            assists.append(assist)
            if i != assist:
                count_assist[i][assist] += 1;
        for i in range(5):
            min_t = update_subpop(list_p[i], offsprings[i], i, assist, r)
            update_archive(list_ac[i], list_p[i])
            print(f"Task {i}: {min_t}  assist task: {assists[i]}")
            min[i].append(min_t)
        assists = []
        offsprings = []
    y = range(1000)
    fig, ax = plt.subplots(5, 2, figsize=(40, 12))
    ax[0, 0].plot(y, min[0])
    ax[0, 0].set_title("Sphere")
    ax[0, 0].set_ylabel('Min')

    ax[0, 1].bar(range(5), count_assist[0], 0.2, alpha=0.6, color='b', label='Sphere')


    ax[1, 0].plot(y, min[1])
    ax[1, 0].set_title('Weierstrass')
    ax[1, 0].set_ylabel('Min')

    ax[1, 1].bar(range(5), count_assist[1], 0.2, alpha=0.6, color='b', label='Weierstrass')


    ax[2, 0].plot(y, min[2])
    ax[2, 0].set_title('Rosenbrock')
    ax[2, 0].set_ylabel('Min')

    ax[2, 1].bar(range(5), count_assist[2], 0.2, alpha=0.6, color='b', label='Rosenbrock')

    ax[3, 0].plot(y, min[3])
    ax[3, 0].set_title('Griewank')
    ax[3, 0].set_ylabel('Min')

    ax[3, 1].bar(range(5), count_assist[3], 0.2, alpha=0.6, color='b', label='Griewank')

    ax[4, 0].plot(y, min[4])
    ax[4, 0].set_title('Rastrigin')
    ax[4, 0].set_xlabel('Generation')
    ax[4, 0].set_ylabel('Min')

    ax[4, 1].bar(range(5), count_assist[4], 0.2, alpha=0.6, color='b', label='Rastrigin')

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.show()

def decode(p, L, U):
    data = []
    for i in p:
        x = []
        for j in i:
            v = L + (U-L)*j
            x.append(v)
        x = np.array(x)
        data.append(x)
    return np.array(data)

if __name__ == "__main__":
    main()



