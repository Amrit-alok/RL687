from Fourier_Transformation import LinearSoftmax
from ReadCSV import ReadBehaviorFile
import numpy as np
from scipy import stats


class PDIS:

    def __init__(self, behavior_file, data_partition):
        self.data_partition = data_partition
        self.behavior_data = ReadBehaviorFile(behavior_file)
        # print("theta_b: ", self.behavior_data.theta_b)
        # print("Policy representation from file: ", self.behavior_data.policy)
        # print("First trajectory",behavior_data.trajectories[1])
        self.theta_b = np.asarray([0.01, -0.01, 1, 1])
        self.delta = 0.05
        self.candidate_pdis = 0
        self.safety_test_size = 200000 - data_partition
        self.safety_pdis = 0
        # self.pdis_for_data = 0
        self.c = 1.028285376929881
        self.candidate_pdis_history = np.array([])
        self.safety_pdis_history = np.array([])
        # self.pdis_history = np.array([])
        self.linear_softmax = LinearSoftmax(self.behavior_data.num_action, self.behavior_data.fourier_order,
                                            self.behavior_data.num_states,
                                            )

    def calculate_PDIS_per_episode(self, behavior_history, theta_e):
        # behavior_history = self.behavior_data.trajectories[1]
        pi_e = 1
        pi_b = 1
        pdis = 0
        policy_representation = []
        for i in range(len(behavior_history) // 3):
            state = float(behavior_history[i * 3])
            action = int(behavior_history[i * 3 + 1])
            reward = float(behavior_history[i * 3 + 2])
            action_probabilities_b = self.linear_softmax.linear_softmax_probs(self.theta_b, state, action)
            action_probabilities_e = self.linear_softmax.linear_softmax_probs(theta_e, state, action)
            pi_b = pi_b * action_probabilities_b
            pi_e = pi_e * action_probabilities_e
            pdis = pdis + (pi_e / pi_b) * reward
            policy_representation.append(action_probabilities_b)
        # print("PDIS:", pdis)
        # print("policy_representation:", policy_representation)
        return pdis

    def generate_pdis_for_candidate(self, theta_e):
        pdis_history = []
        count = 0
        for history in self.behavior_data.trajectories[1:self.data_partition]:
            # print("history", history)
            pdis_per_episdode = self.calculate_PDIS_per_episode(history, theta_e)
            # print(pdis_per_episdode)
            # np.append(self.candidate_pdis_history,pdis_per_episdode)
            pdis_history.append(pdis_per_episdode)
            # count += 1
            # if(count%5000==0):
            #     print("pdis_per_episdode",pdis_per_episdode)
        self.candidate_pdis_history = np.array(pdis_history)

    def estimate_J_theta_candidate(self, theta_e):
        # print("theta_e",theta_e)
        self.generate_pdis_for_candidate(theta_e)
        # print("History of J",self.candidate_pdis_history)
        # print("Max PDIS",np.max(self.pdis_history))
        self.candidate_pdis = np.mean(self.candidate_pdis_history)
        # print("J_e:", self.pdis_for_data)
        # return self.pdis_for_data
        return self.candidate_pdis

    def upper_bound(self, theta_e):
        # safety_test_size = 200000 - self.data_partition
        safety_test_size = np.sqrt(self.safety_test_size)
        upper_limit = self.estimate_J_theta_candidate(theta_e) - 2 * (
            np.divide(self.standard_deviation_pdis(), safety_test_size)) * stats.t.ppf(1 - self.delta,
                                                                                       safety_test_size - 1)
        print("Upper_Limit is", upper_limit)
        if upper_limit >= self.c:
            return upper_limit
        else:
            return -100000

    # def student_t_test(self):
    #     # print("J_Theta", self.pdis_for_data)
    #     return self.candidate_pdis - (
    #             self.standard_deviation_pdis() / np.sqrt(len(self.candidate_pdis_history)) * stats.t.ppf(
    #         1 - self.delta, len(self.candidate_pdis_history) - 1))

    def standard_deviation_pdis(self):
        pdis_list = self.candidate_pdis_history
        pdis_for_data = self.candidate_pdis
        return np.sqrt(np.sum((pdis_list - pdis_for_data) ** 2) / (len(pdis_list) - 1))

    #

    def generate_pdis_for_safety(self, theta_e):
        # print("Theta_e for safety", theta_e)
        pdis_history_safety = []
        start = self.data_partition
        end = self.data_partition + self.safety_test_size
        # count = 1
        # print(start,  end)
        for history in self.behavior_data.trajectories[start:end]:
            # print("history", history, "End",end)
            pdis_per_episdode = self.calculate_PDIS_per_episode(history, theta_e)
            #print("pdis_per_episdode",pdis_per_episdode)
            # np.append(self.candidate_pdis_history,pdis_per_episdode)
            pdis_history_safety.append(pdis_per_episdode)
            # count += 1
            # if(count%5000==0):
            #     print("pdis_per_episdode",pdis_per_episdode)
        # print(count)
        self.safety_pdis_history = np.array(pdis_history_safety)

    def estimate_J_theta_safety(self, theta_e):
        self.generate_pdis_for_safety(theta_e)
        # print("safety_pdis_history",self.safety_pdis_history)
        self.safety_pdis = np.mean(self.safety_pdis_history)
        return self.safety_pdis

    def standard_deviation_safety(self):
        pdis_list = self.safety_pdis_history
        pdis_for_safety = self.safety_pdis
        return np.sqrt(np.sum((pdis_list - pdis_for_safety) ** 2) / (len(pdis_list) - 1))

    def execute_safety_test(self, theta_e):
        print("safety theta",theta_e)
        self.estimate_J_theta_safety(theta_e)
        b = self.safety_pdis - (
                np.divide(self.standard_deviation_safety(), np.sqrt(self.safety_test_size)) * stats.t.ppf(
            1 - self.delta, self.safety_test_size - 1))
        if b >= self.c:
            print("safety test passed")
        else:
            print("safety test failed")

pdis = PDIS("data.csv", 50000)
print("Student_T_Test", pdis.estimate_J_theta_candidate([1.30716094, -4.01400105, -2.39934448,  3.18802846]))

# print("Student_T_Test", pdis.estimate_J_theta_candidate([-162.42145502, 568.02024562, -625.48083063, -30.46094259]))

# # # print("Student_T_Test", pdis.student_t_test([1.26832255, -0.28310174, -1.9134403, -0.57913468]))
# # print("Student_T_Test", pdis.estimate_J_theta([0.9374667, 0.88213681, -0.74731632, -0.36047931]))
# print("Student_T_Test", pdis.estimate_J_theta([ 4.39320166,  5.5556872,  -5.1565573,  -6.8313533]))
# # # print("Student_T_Test", pdis.student_t_test([0.01, -0.01, 1, 1]))
