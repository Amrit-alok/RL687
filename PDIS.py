from Fourier_Transformation import LinearSoftmax
from Fourier_base_transformation import FourierTransformation
from ReadCSV import ReadBehaviorFile
import numpy as np
from scipy import stats


class PDIS:

    def __init__(self, behavior_file):
        self.behavior_data = ReadBehaviorFile(behavior_file)
        print("theta_b: ", self.behavior_data.theta_b)
        print("Policy representation from file: ", self.behavior_data.policy)
        # print("First trajectory",behavior_data.trajectories[1])
        self.theta_b = np.asarray([0.01, -0.01, 1, 1])
        # self.theta_b = [[0.01, -0.01, ], [1, 1]]
        # self.theta_e = np.asarray(theta_e)
        self.delta = 0.05
        self.pdis_for_data = 0
        self.pdis_history = np.array([])
        self.linear_softmax = LinearSoftmax(self.behavior_data.num_action, self.behavior_data.fourier_order,
                                            self.behavior_data.num_states,
                                            )

    def calculate_PDIS_per_episode(self, behavior_history, theta_e):
        # behavior_history = self.behavior_data.trajectories[1]
        pi_e = 1
        pi_b = 1
        pdis = 0
        policy_representation = []
        # linear_softmax_b = LinearSoftmax(self.behavior_data.num_action, self.behavior_data.fourier_order,
        #                                  self.behavior_data.num_states,
        #                                  self.theta_b)
        # linear_softmax_e = LinearSoftmax(self.behavior_data.num_action, self.behavior_data.fourier_order,
        #                                  self.behavior_data.num_states,
        #                                  theta_e)
        for i in range(len(behavior_history) // 3):
            state = float(behavior_history[i * 3])
            action = int(behavior_history[i * 3 + 1])
            ##print("action:",action)
            reward = float(behavior_history[i * 3 + 2])
            '''
            linear_softmax_b = FourierTransformation(self.theta_b, self.behavior_data.num_action,
                                                     self.behavior_data.num_states, self.behavior_data.fourier_order,
                                                     action)
            linear_softmax_e = FourierTransformation(self.theta_e, self.behavior_data.num_action,
                                                     self.behavior_data.num_states, self.behavior_data.fourier_order,
                                                     action)
            '''

            action_probabilities_b = self.linear_softmax.linear_softmax_probs(self.theta_b, state, action)
            action_probabilities_e = self.linear_softmax.linear_softmax_probs(theta_e, state, action)
            #
            # action_probabilities_b = linear_softmax_b.transformation(state)
            # action_probabilities_e = linear_softmax_e.transformation(state)
            pi_b = pi_b * action_probabilities_b
            pi_e = pi_e * action_probabilities_e
            pdis = pdis + (pi_e / pi_b) * reward
            policy_representation.append(action_probabilities_b)
        # print("PDIS:", pdis)
        print("policy_representation:", policy_representation)
        return pdis

    def calculate_PDIS_Data(self, theta_e):
        pdis_history = []
        count = 0
        for history in self.behavior_data.trajectories[1:3]:
            # print("history", history)
            pdis_per_episdode = self.calculate_PDIS_per_episode(history, theta_e)
            pdis_history.append(pdis_per_episdode)
            count += 1
            # if(count%5000==0):
            #     print("pdis_per_episdode",pdis_per_episdode)
        self.pdis_history = np.array(pdis_history)

    def estimate_J_theta(self, theta_e):
        # print("theta_e",theta_e)
        self.calculate_PDIS_Data(theta_e)
        # print("History of J",self.pdis_history)
        # print("Max PDIS",np.max(self.pdis_history))
        self.pdis_for_data = np.mean(self.pdis_history)
        # print("J_e:", self.pdis_for_data)
        return self.pdis_for_data

    def student_t_test(self, theta_e):
        self.estimate_J_theta(theta_e)
        print("J_Theta", self.pdis_for_data)
        return self.pdis_for_data - (
                self.standard_deviation_pdis() / np.sqrt(len(self.pdis_history)) * stats.t.ppf(
            1 - self.delta, len(self.pdis_history) - 1))

    def standard_deviation_pdis(self):
        pdis_list = self.pdis_history
        pdis_for_data = self.pdis_for_data
        # print(pdis_for_data)
        # print()
        return np.sqrt(np.sum((pdis_list - pdis_for_data) ** 2) / (len(pdis_list) - 1))

pdis = PDIS("data.csv")
print("Student_T_Test", pdis.student_t_test([1.26832255, -0.28310174, -1.9134403,  -0.57913468]))
