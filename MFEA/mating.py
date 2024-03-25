import numpy as np

rmp = 0.3

def assortative_mating(cur_pop, skill_factor):
    offsprings = []
    offsprings_skill_factor = []
    for i in range(25):
        rnd = np.random.rand(1)[0]
        random_indices = np.random.choice(np.arange(len(cur_pop)), size=2, replace=False)
        if skill_factor[random_indices[0]] == skill_factor[random_indices[1]] or rnd < rmp:
            offspring = np.concatenate((cur_pop[random_indices[0]][:5], cur_pop[random_indices[1]][-5:]))
            offsprings.append(offspring)
            random_skill_factor = np.random.rand(1)[0]
            if random_skill_factor < 0.5:
                offsprings_skill_factor.append(skill_factor[random_indices[0]])
            else:
                offsprings_skill_factor.append(skill_factor[random_indices[1]])
        else:
            ran_idx0 = np.random.choice(np.arange(len(cur_pop[random_indices[0]])), size=1, replace=False)[0]
            ran_idx1 = np.random.choice(np.arange(len(cur_pop[random_indices[1]])), size=1, replace=False)[0]
            ran_value = np.random.rand(2)
            offspring0 = cur_pop[random_indices[0]].copy()
            offspring1 = cur_pop[random_indices[1]].copy()
            offspring0[ran_idx0] = ran_value[0]
            offspring1[ran_idx1] = ran_value[1]
            offsprings.append(offspring0)
            offsprings_skill_factor.append(skill_factor[random_indices[0]])
            offsprings.append(offspring1)
            offsprings_skill_factor.append(skill_factor[random_indices[1]])
    return np.array(offsprings), np.array(offsprings_skill_factor)