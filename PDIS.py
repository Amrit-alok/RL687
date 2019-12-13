from Fourier_Transformation import LinearSoftmax
from ReadCSV import ReadBehaviorFile
import numpy as np
from scipy import stats


class PDIS:

    @property
    def pi_b(self):
        return self._pi_b

    def __init__(self, behavior_file, start_index, end_index):
        data_partition = end_index - start_index
        self.start_index = start_index
        self.end_index = end_index
        self.data_partition = int(data_partition * 0.6)
        print("candidate_data", self.data_partition)
        self.behavior_data = ReadBehaviorFile(behavior_file, start_index, end_index)
        self.theta_b = np.asarray([0.01, -0.01, 1, 1])
        self.delta = 0.05  # 0.001
        self.candidate_pdis = 0
        self.safety_test_size = int(data_partition * 0.4)
        print("safety_data", self.safety_test_size)
        self.safety_pdis = 0
        self.c = 2 * 1.028285376929881
        self.candidate_pdis_history = np.array([])
        self.safety_pdis_history = np.array([])
        self.linear_softmax = LinearSoftmax(self.behavior_data.num_action, self.behavior_data.fourier_order,
                                            self.behavior_data.num_states,
                                            )

    def calculate_pi_b(self):
        behavior_policy = [[]]
        print("Pi_b is being calculated")
        # for trajectory in self.behavior_data.trajectories[self.start_index+1:self.end_index + 1]:
        for trajectory in self.behavior_data.trajectories[1:self.data_partition + self.safety_test_size + 1]:
            pi_b = []
            probabilities = 1
            for i in range(len(trajectory) // 3):
                state = float(trajectory[i * 3])
                action = int(trajectory[i * 3 + 1])
                action_probability = self.linear_softmax.linear_softmax_probs(self.theta_b, state, action)
                probabilities = probabilities * action_probability
                pi_b.append(probabilities)
            behavior_policy.append(pi_b)
            # print("For trajectory", trajectory, ":pi_b is: ", pi_b)
        self.pi_b = np.array(behavior_policy[1:])
        print("pi_b is calculated")
        print("size of pi_b:", len(self.pi_b))

    def calculate_PDIS_per_episode(self, behavior_history, theta_e, pi_b):
        pi_e = 1
        pdis = 0
        # print("pi_b", pi_b)
        for i in range(len(behavior_history) // 3):
            state = float(behavior_history[i * 3])
            action = int(behavior_history[i * 3 + 1])
            reward = float(behavior_history[i * 3 + 2])
            action_probabilities_e = self.linear_softmax.linear_softmax_probs(theta_e, state, action)
            pi_e = pi_e * action_probabilities_e
            pdis = pdis + (pi_e / pi_b[i]) * reward
        return pdis

    def generate_pdis_for_candidate(self, theta_e):
        pdis_history = []
        # self.calculate_pi_b()
        # print(self.pi_b)
        # count = self.start_index
        count = 0
        # for history in self.behavior_data.trajectories[self.start_index+1:self.start_index + self.data_partition + 1]:
        for history in self.behavior_data.trajectories[1:self.data_partition + 1]:
            pdis_per_episdode = self.calculate_PDIS_per_episode(history, theta_e, self.pi_b[count])
            pdis_history.append(pdis_per_episdode)
            # print("in Candidate for trajectory:", history, "pi_b:", self.pi_b[count])
            count += 1
        self.candidate_pdis_history = np.array(pdis_history)
        # print("candidate_data_processed:", count + 1)
        # print("candidate_pdis_len", len(self.candidate_pdis_history))

    def estimate_J_theta_candidate(self, theta_e):
        self.generate_pdis_for_candidate(theta_e)
        self.candidate_pdis = np.mean(self.candidate_pdis_history)
        return self.candidate_pdis

    def upper_bound(self, theta_e):
        safety_test_size = self.safety_test_size
        upper_limit = self.estimate_J_theta_candidate(theta_e) - (2 * (
            np.divide(self.standard_deviation_pdis(), np.sqrt(safety_test_size))) * stats.t.ppf(1 - self.delta,
                                                                                                safety_test_size - 1))
        if upper_limit >= self.c:
            return upper_limit
            # return self.candidate_pdis
        else:
            return -100000 + upper_limit

    def standard_deviation_pdis(self):
        return np.std(self.candidate_pdis_history)
        ##pdis_candidate_list = self.candidate_pdis_history
        # pdis_for_candidate_data = self.candidate_pdis
        # return np.sqrt(np.sum((pdis_candidate_list - pdis_for_candidate_data) ** 2) / (len(pdis_candidate_list) - 1))

    def generate_pdis_for_safety(self, theta_e):
        # print("Theta_e for safety", theta_e)
        pdis_history_safety = []
        start = self.data_partition + 1
        # start = self.start_index+self.data_partition + 1
        end = self.data_partition + self.safety_test_size + 1
        # end = self.end_index + 1
        count = start
        for history in self.behavior_data.trajectories[start:end]:
            # print("count is:", count)
            # print("In safety for trajectory",history,"pi_b is",self.pi_b[count-1])
            pdis_per_episdode = self.calculate_PDIS_per_episode(history, theta_e, self.pi_b[count - 1])
            pdis_history_safety.append(pdis_per_episdode)
            count += 1
        self.safety_pdis_history = np.array(pdis_history_safety)
        # print("length of safety pdis", len(self.safety_pdis_history))

    def estimate_J_theta_safety(self, theta_e):
        self.generate_pdis_for_safety(theta_e)
        self.safety_pdis = np.mean(self.safety_pdis_history)
        return self.safety_pdis

    def standard_deviation_safety(self):
        return np.std(self.safety_pdis_history)
        # pdis_list = self.safety_pdis_history
        # pdis_for_safety = self.safety_pdis
        # return np.sqrt(np.sum((pdis_list - pdis_for_safety) ** 2) / (len(pdis_list) - 1))

    def execute_safety_test(self, theta_e, i):
        print("safety theta for row:", i, " ", theta_e)
        print("Executing safety test")
        self.estimate_J_theta_safety(theta_e)
        b = self.safety_pdis - (
                np.divide(self.standard_deviation_safety(), np.sqrt(self.safety_test_size)) * stats.t.ppf(
            1 - self.delta, self.safety_test_size - 1))

        print("B", b)
        print("Theta_e_estimate_safety:", self.safety_pdis)
        if b >= self.c:
            print("safety test passed")
            # np.savetxt("policy"+str(i)+".csv", theta_e)
        else:
            print("safety test failed")

    @pi_b.setter
    def pi_b(self, value):
        self._pi_b = value


pdis = PDIS("data.csv", 0, 200000)
pdis.calculate_pi_b()
# print("Student_T_Test", pdis.estimate_J_theta_candidate([-4.52376687, 2.71429282, -9.23526859, 8.41318765]))
array = [[-0.10762815309384006, -1.2452739068421796, 4.3334647974302, 5.74272207055282],
         [-2.7823139556145433, 7.454027649919496, 1.7291821211005458, 3.4505311056372876],
         [1.4820065291022984, 0.5042800569790986, 4.807644704460583, 2.701641742682958],
         [1.7006014590363434, 2.62234732377758, -2.1272516963916517, -7.262709031806967],
         [3.0077278471947304, 3.6081146060861955, 1.8005828145442022, -4.889773546652831],
         [-1.3784212222292367, 6.514794160259696, -1.7633389324828816, 3.4928465464615632],
         [6.618026184211759, 0.805092321302032, -0.6873709348248755, -4.396831876440591],
         [-2.021055469890009, -7.250176079261984, 5.074797886644944, 1.5606362040697497],
         [12.02490891568059, 1.070288629051287, -4.954091845160269, -6.453114646396339],
         [1.135169705289284, 5.745719333611262, -3.8305434928586957, -15.565292397656622],
         [11.905163350328927, 3.54170779190003, 4.6599854751227205, -14.852539766124881],
         [-2.285573782421965, 8.504044986075149, -1.6264210505603285, -4.161062017433187],
         [5.957062787959677, -1.6154173636454896, 4.720980646791338, -11.799010309917055],
         [-4.001734928306278, 6.620675331293537, -2.1189016386380986, -0.5796534397818558],
         [6.726434166502398, -4.759624114146181, -4.295342945235014, -2.8856716892642433],
         [5.602039025034188, 5.194600429638381, 1.3468520644019975, -8.825668023950996],
         [2.6538423730246556, -0.8762611058337857, 3.844926764269517, -9.581408883926478],
         [9.25762197333665, 9.534304926491295, -3.176805412316332, -35.762143568851656],
         [1.1240420160281612, 7.020533430916377, 5.420583668428572, -12.801923490818421],
         [9.65350131641951, 6.736208520351031, -4.074467641365676, -22.46085799329861],
         [15.799083926454188, 12.633420288332385, -8.220521919913454, -17.43925102308195],
         [-3.102781516207629, 3.632099240177085, -0.6681650207470384, -7.028406354731432],
         [1.575393405535725, 1.351230220430962, -7.501318049979358, -13.90786550506647],
         [-3.2389432183858657, 2.819223788283684, -0.27121375880476273, -6.471163007241903],
         [13.061568411880113, 0.6209470614178212, -13.65865745719556, -38.16859561134892],
         [36.26973093906864, 15.27932382017719, -0.4629724812126774, -42.578185486111884],
         [-5.049320019892894, -3.7324977859837425, 7.495037416229109, -14.044243417566538],
         [0.4479955569930727, 8.674673194908252, -3.5504266077790536, -34.34246475685558],
         [2.3304554068896737, 8.811771977514093, 5.240803232191253, -2.968287633788737],
         [1.330731643604243, 10.415396514952311, -24.408492919033666, -25.2476054923147],
         [-1.9459115424092648, -14.606866060429647, 4.8897857906497135, -22.2126942420014],
         [10.492783490119859, -6.222608163358741, -1.0276237545022027, -20.356812865900142],
         [-3.3508392336950212, 21.483531641377347, 1.093485103497958, -20.123995975597598],
         [23.292087865193864, 30.014440614812578, -2.1349739003164525, -61.280548013551375],
         [28.959199893502706, 16.382850020511206, 3.8139000257109217, -53.636339407417566],
         [11.75609817906677, 36.26437728505952, -29.902341994312174, -52.29363787071719],
         [5.922168891305856, 8.680078341619037, -6.79394338809745, -53.112650215298856],
         [-9.850944049960551, -2.9896378121232647, -3.73956873992256, -3.6737638793407115],
         [37.49075417451109, 37.011210714805244, 1.1250204527840317, -60.09699477536276],
         [12.21180711568584, 17.155732571370173, -6.687468924116443, -54.39517791422097],
         [11.15662500695909, 14.469112474577862, -5.65774665230228, -54.66766150004864]]
i = 1
for theta_e in array:
    pdis.execute_safety_test(theta_e, i)
    i += 1
# pdis = PDIS("data.csv", 20,30)
# pdis.calculate_pi_b()
# print("Student_T_Test", pdis.estimate_J_theta_candidate([-4.52376687, 2.71429282, -9.23526859, 8.41318765]))
# print("safet_test", pdis.execute_safety_test([2.06836668, -1.40374511, -2.26690564, -0.70637846]))
# print("Student_T_Test", pdis.estimate_J_theta_candidate([1.30716094, -4.01400105, -2.39934448,  3.18802846]))

# print("Student_T_Test", pdis.estimate_J_theta_candidate([-162.42145502, 568.02024562, -625.48083063, -30.46094259]))

# # # print("Student_T_Test", pdis.student_t_test([1.26832255, -0.28310174, -1.9134403, -0.57913468]))
# # print("Student_T_Test", pdis.estimate_J_theta([0.9374667, 0.88213681, -0.74731632, -0.36047931]))
# print("Student_T_Test", pdis.estimate_J_theta([ 4.39320166,  5.5556872,  -5.1565573,  -6.8313533]))
# # # print("Student_T_Test", pdis.student_t_test([0.01, -0.01, 1, 1]))
