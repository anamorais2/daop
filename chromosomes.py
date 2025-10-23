import random


def random_da_func(da_funcs_number):
    # random data augmentation function
    def get_random_da_func():
        return random.randint(0,da_funcs_number-1)
    
    return get_random_da_func

def random_pr():
    # random parameter
    return round(random.random(),2)

def random_pr_gaussian(sigma):
    # random parameter but based on gaussian distribution
    def get_random_pr_gaussian(mu):
        return max(0, min(1, round(random.gauss(mu, sigma),2)))
    
    return get_random_pr_gaussian


def create_chromosome_2_levels(create_da_func, create_pr, n_pr):
    # create chromosome with 2 levels (da_func and p0 (prob), p1, p2, p3 of each da_func)
    def get_chromosome_2_levels():
        return [
            create_da_func(), # da_func
            [create_pr() for _ in range(n_pr+1)]  # p0 (probability of applying da_func), p1, p2, p3
        ] 
    
    return get_chromosome_2_levels

def mutate_chromosome_2_levels(da_func_mutation, pr_mutation, n_pr):
    # mutate entire chromosome with 2 levels (da_func and p0 (prob), p1, p2, p3 of each da_func)
    def mutate_chromosome(chromosome, config):
        return [
            da_func_mutation(), # da_func
            [pr_mutation(chromosome[1][i]) for i in range(n_pr+1)]  # p0 (probability of applying da_func), p1, p2, p3
        ]
    
    return mutate_chromosome