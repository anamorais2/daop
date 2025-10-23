import copy
import random
import chromosomes


def mutate_remove_change_add_seq(remove_rate, mutation_rate, add_rate):
    def mutate(parent, config):
        new_individual = copy.deepcopy(parent)

        # remove_rate probability of removing a chromosome
        if len(new_individual) > 1 and random.random() < remove_rate:
            new_individual.pop(random.randint(0, len(new_individual) - 1))

        # mutation_rate probability of mutating (changing) each chromosome
        for i in range(len(new_individual)):
            if random.random() < mutation_rate:
                # mutate data augmentation function
                new_individual[i][0] = config['da_func_mutation']()
            else:   
                # if da_func was changed, no need to mutate parameters (the old values don't have the 
                # same meaning in the new function, almost as if it was random)
                for j in range(len(new_individual[i][1])):
                    if random.random() < mutation_rate:
                        # mutate p0 (probability of applying da_func), p1, p2, p3
                        new_individual[i][1][j] = config['pr_mutation'](new_individual[i][1][j])

        # add_rate probability of adding a chromosome
        if len(new_individual) < config['max_chromosomes'] and random.random() < add_rate:
            new_individual.insert(random.randint(0, len(new_individual)),config['create_chromosome']())

        return new_individual
            
    return mutate


def mutate_remove_change_add_1mut(chromosome_mutation_rate, chromosome_mutation_func):
    def mutate(parent, config):
        new_individual = []
        i = 0
        total_len = len(parent)
        while i < total_len:
            # chromosome_mutation_rate probability of mutating each chromosome of the parent
            if random.random() < chromosome_mutation_rate:
                add_change_rem = random.randint(0, 2)
                ADD_CHROMOSOME = 0
                CHANGE_CHROMOSOME = 1
                # REMOVE_CHROMOSOME = 2
                if add_change_rem == ADD_CHROMOSOME:
                    new_individual.append(
                        config['create_chromosome']()
                    )
                    i -= 1
                elif add_change_rem == CHANGE_CHROMOSOME:
                    new_individual.append(
                        chromosome_mutation_func(parent[i], config)
                    )
                # if REMOVE_CHROMOSOME, do nothing, simply don't append the corresponding chromosome
            else:
                new_individual.append(parent[i])
            i += 1

        return new_individual
            
    return mutate
