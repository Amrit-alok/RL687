import numpy as np
from PDIS import PDIS
from es import CMAES


def problem4():
    data_parition = 200000
    pdis = PDIS(behavior_file="data.csv", data_partition=data_parition)
    pdis.calculate_pi_b()

    for trials in range(10):
        NPARAMS = 4  # make this a 100-dimensinal problem.
        NPOPULATION = 50  # use population size of 101.
        MAX_ITERATION = 6
        cmaes = CMAES(NPARAMS,
                      popsize=NPOPULATION,
                      weight_decay=0.0,
                      sigma_init=2.0
                      )

        history = []
        print("candidate_data: ",data_parition,"Population: ",NPOPULATION, "Max Iteration: ",MAX_ITERATION)
        for j in range(MAX_ITERATION):
            solutions = cmaes.ask()
            fitness_list = np.zeros(cmaes.popsize)
            for i in range(cmaes.popsize):
                fitness_list[i] = pdis.upper_bound(solutions[i])
            cmaes.tell(fitness_list)
            result = cmaes.result()  # first element is the best solution, second element is the best fitness
            history.append(result[1])
            if (j + 1) % 2 == 0:
                print("fitness at iteration", (j + 1), result[1])
        print("local optimum discovered by solver:\n", result[0])
        print("fitness score at this local optimum:", result[1])
        # return history
        pdis.execute_safety_test(result[0],trials)


problem4()