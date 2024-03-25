import numpy as np
import  random
from mating import assortative_mating

kp_dim = 10
qap_dim = 10
capacity = 20
kp_v = np.array([10, 5, 12, 7, 9, 15, 3, 6, 8, 10])
kp_w = np.array([5, 3, 8, 4, 6, 10, 2, 5, 7, 6])
kp_eff = np.array(kp_v/kp_w)
print(kp_eff)

qap_d_matrix = np.random.randint(1, 10, size=(qap_dim, qap_dim))
np.fill_diagonal(qap_d_matrix, 0)
qap_f_matrix = np.random.randint(1, 10, size=(qap_dim, qap_dim))
np.fill_diagonal(qap_f_matrix, 0)
def init():
    cur_pop = np.random.rand(10, 10)
    rank_kp = kp_rank(cur_pop)
    rank_qap = qap_rank(cur_pop)
    skill_factor = []
    for i in range(10):
        if rank_kp[i] < rank_qap[i]:
            skill_factor.append(0)
        else:
            skill_factor.append(1)
    return cur_pop, np.array(skill_factor)
def kp_cost(p):
    result = []
    p = Danzig_Reassign_KP(p)
    for i in p:
        x = kp_decode(i)
        Knap_sack = 0
        for j in np.where(x == 1)[0]:
            Knap_sack -= kp_v[j]
        result.append(Knap_sack)
    return np.array(result)

def Danzig_Reassign_KP(p):
    new_p = []
    for i in p:
        x = kp_decode(i)
        weight = 0
        idx_item_in_bag = np.where(x == 1)[0]
        eff = kp_eff[idx_item_in_bag]
        rank_eff = np.argsort(eff)
        for j in idx_item_in_bag:
            weight += kp_w[j]
        new_i = i.copy()
        k = 0
        while weight > capacity:
            new_i[idx_item_in_bag[rank_eff[k]]] = random.random() * 0.5
            weight -= kp_w[idx_item_in_bag[rank_eff[k]]]
            k += 1
        new_p.append(new_i)
    return np.array(new_p)


def qap_cost(p):
    result = []
    for i in p:
        x = qap_decode(i);
        total_cost = 0;
        for i in range(qap_dim):
            for j in range(qap_dim):
                facility1 = x[i]
                facility2 = x[j]
                location1 = i
                location2 = j

                total_cost += qap_f_matrix[facility1][facility2]*qap_d_matrix[location1][location2]
        result.append(total_cost)
    return np.array(result)


def kp_decode(p):
    data = []
    for i in p:
        if i < 0.5:
            data.append(0)
        else:
            data.append(1)
    return np.array(data)

def qap_decode(p):
    return np.argsort(p)

def kp_rank(p):
    return np.argsort(np.argsort(kp_cost(p)))

def qap_rank(p):
    return np.argsort(np.argsort(qap_cost(p)))

def update_scalar_fitness(p, s):
    kp_pop = p[np.where(s == 0)[0]]
    rank_kp = kp_rank(kp_pop)
    scalar_fitness_kp = [1/(i+1) for i in rank_kp]

    qap_pop = p[np.where(s == 1)[0]]
    rank_qap = qap_rank(qap_pop)
    scalar_fitness_qap = [1/(i+1) for i in rank_qap]
    scalar_fitness = np.random.rand(p.shape[0])
    scalar_fitness[np.where(s == 0)[0]] = scalar_fitness_kp
    scalar_fitness[np.where(s == 1)[0]] = scalar_fitness_qap
    filter_indices = np.argsort(scalar_fitness)[:25]
    fittest_pop = np.delete(p, filter_indices, axis=0)
    fittest_skill_factor = np.delete(s, filter_indices)
    return fittest_pop, fittest_skill_factor

def main():
    cur_pop, skill_factor = init()
    generation = 100
    for i in range(generation):
        offspring_pop, offspring_skill_factor = assortative_mating(cur_pop, skill_factor)
        intermediate_pop = np.vstack((cur_pop, offspring_pop))
        intermediate_skill_factor = np.concatenate((skill_factor, offspring_skill_factor))
        cur_pop, skill_factor = update_scalar_fitness(intermediate_pop, intermediate_skill_factor)
        kp_pop = cur_pop[np.where(skill_factor == 0)[0]]
        qap_pop = cur_pop[np.where(skill_factor == 1)[0]]
        print("KP: ", kp_cost(kp_pop).min())
        print("QAP: ", qap_cost(qap_pop).min())


if __name__ == "__main__":
    main()



