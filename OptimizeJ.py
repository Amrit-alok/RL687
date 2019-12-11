import numpy as np
from PDIS import PDIS
from matplotlib import pyplot as plot

from cem import CEM
from es import CMAES


def problem4():
        """
        Repeat the previous question, but using the cross-entropy method on the
        cart-pole domain. Notice that the state is not discrete, and so you cannot
        directly apply a tabular softmax policy. It is up to you to create a
        representation for the policy for this problem. Consider using the softmax
        action selection using linear function approximation as described in the notes.
        Report the same quantities, as well as how you parameterized the policy.
        """
        # TODO
        # num_actions = 2
        # num_states = 1
        # theta = np.random.rand(num_actions * (1 + 1) ** num_states)
        # trials = 1
        # sigma = 10
        pdis = PDIS(behavior_file="data.csv")
        # episodes = 20
        # pop_size = 50
        # num_elite = 10
        # epsilon = 1.5
        # cem = CEM(theta, sigma, pop_size, num_elite, episodes, pdis.estimate_J_theta, epsilon)
        # total_return_trial = []
        # print("Cartpole_CEM")
        # for trial in range(trials):
        #     print("trial", trial)
        #     cem.reset()
        #     for itr in range(50):
        #         print("hello")
        #         cem.train()
        # print("optimimized theta:",cem._theta)
        NPARAMS = 4  # make this a 100-dimensinal problem.
        NPOPULATION = 100  # use population size of 101.
        MAX_ITERATION = 4
        cmaes = CMAES(NPARAMS,
                      popsize=NPOPULATION,
                      weight_decay=0.0,
                      sigma_init=0.5
                      )
        history = []
        for j in range(MAX_ITERATION):
            solutions = cmaes.ask()
            fitness_list = np.zeros(cmaes.popsize)
            for i in range(cmaes.popsize):
                fitness_list[i] = pdis.estimate_J_theta(solutions[i])
            cmaes.tell(fitness_list)
            result = cmaes.result()  # first element is the best solution, second element is the best fitness
            history.append(result[1])
            if (j + 1) % 100 == 0:
                print("fitness at iteration", (j + 1), result[1])
        print("local optimum discovered by solver:\n", result[0])
        print("fitness score at this local optimum:", result[1])
        return history

problem4()

