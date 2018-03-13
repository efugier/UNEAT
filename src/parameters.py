
# Parameters

nb_input = 2
nb_output = 1


# Genetic Algorithm parameters

# coefficients for the distance calculation
disjoint_coeff = 1
recursive_disjoint_coeff = 1
average_weight_coeff = 0.4
average_recursive_weight_coeff = 1

same_species_threshold = 3

squaring_factor = 1  # high value => all or nothing
scaling_factor = 1


crossover_rate = 0.75
elimination_rate = 0.4
max_stagnation = 6

weight_mutation_proba = 0.8
uniform_perturbation_proba = 0.9

new_connexion_proba = 0.05
new_recursive_connexion_proba = 0
force_input_proba = 0.1

new_neuron_proba = 0.03  # 0.3 if larger population
