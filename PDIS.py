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
        self.delta = 0.001
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
            # return upper_limit
            return self.candidate_pdis
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

    def execute_safety_test(self, theta, itr):
        print("Executing safety theta for row:", itr, " ", theta)
        self.estimate_J_theta_safety(theta)
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


array = [
    [-1.48810483, -5.8838954, -6.66486254, 2.50146831],
    [4.04246036, -4.11768224, -1.05767587, 4.08253872],
    [-3.79293304, -1.92418096, -8.56187103, 5.43071975],
    [2.14252243, -4.820573, -2.89702609, 3.2813018],
    [4.12973742, -3.45175617, -0.65938406, 4.04142912],
    [2.1619701, -2.45386902, -1.03488262, 4.34338135],
    [2.22497536, -22.64193402, -15.81590115, 10.73880119],
    [1.13819907, -4.85300822, -2.16992484, 1.14013321],
    [2.20230796, -4.14083396, -2.24512397, 4.02302285],
    [0.31500172, 8.45417112, -14.55203614, -13.78692092],
    [25.52102725, 42.39934163, -19.87464904, -49.36815996],
    [6.08451293, 21.21244758, -14.02773664, -17.16182465],
    [-12.20894002, 13.98214037, -34.35392966, -23.45040965],
    [-17.35728131, -3.54574772, -40.00763546, -40.36229815],
    [39.86646796, 6.43106706, -1.9504568, -76.61021066],
    [4.69316609, 0.86241457, -5.32928995, -12.38306742],
    [20.96971486, 23.25397895, -19.00666044, -33.50128026],
    [10.22528905, 8.28499955, -0.93434349, -9.96264375],
    [0.91229022, 18.48903157, -22.46971372, -21.14665856],
    [4.1653161, 14.7216328, -5.1545595, -1.4765127],
    [10.56013114, -0.36749792, -0.78660299, -19.85134882],
    [9.19329064, 24.1841517, -8.77129445, -1.78425686],
    [15.29551905, -10.76655494, -32.21698019, -5.37469248],
    [9.09883384, 25.43011971, -8.00497706, -3.59536647],
    [1.90084407, -3.95114143, -3.54221692, 3.44881259],
    [4.21605397, -6.8637894, -0.81095486, 1.36656306],
    [4.65308424, -6.48017337, -0.50253372, 1.80624385],
    [0.54197932, -1.72560242, -4.0787447, 5.41850286],
    [2.670076, -1.66382743, -2.69859878, 5.23385109],
    [17.03281298, 17.29377882, -4.08675289, -11.70947292],
    [0.42092926, -9.01332899, -5.23542439, -1.66880403],
    [1.58155266, -10.25664505, -2.99872638, -3.19882466],
    [5.73811512, 0.05123619, 1.20104827, 7.07542634],
    [16.40678705, 24.0770763, -4.68626395, -4.38386291],
    [10.94115052, 23.01397418, -30.76391804, -31.89141458],
    [9.06756454, 22.81374096, -20.56410856, -33.20978888],
    [14.19659894, 12.95352649, -3.4321726, -17.10323524],
    [4.10708457, 19.01849009, -14.21709444, -7.95658671],
    [8.62979349, 0.19344867, -2.76030951, -14.55573687],
    [17.84488302, 23.56938884, -10.10120127, -14.20852954],
    [-0.33592529, 17.67492249, -19.22236466, -9.14605575],
    [10.12792368, 6.43585383, -17.20993125, -3.23482918],
    [7.15753137, 13.20836049, -10.93251274, -12.2448044],
    [4.42824286, 12.17904419, -9.96959445, -16.16016678],
    [-4.19501593, 20.24324947, -18.31270114, -1.56635012],
    [29.17065508, 17.05351544, 1.83221904, -18.54123141],
    [3.56533486, 13.01445372, -15.60380559, -14.32547075],
    [12.21969705, 26.63799555, -22.10912233, -37.75283195],
    [22.87182672, 8.78230839, 8.7752644, -14.35973983],
    [7.92946121, 40.47337606, -15.95149641, 8.3654024],
    [0.12604868, -4.24075395, -5.29104123, 4.65736742],
    [15.38384309, 28.83227474, -15.20398549, -20.87668876],
    [9.9217413, 9.30686695, -0.50853114, -4.7031652],
    [8.57510479, 13.09113914, -9.67246697, -22.43179039],
    [19.98814066, 39.28638928, -28.03951572, -24.64307934],
    [9.98626059, 27.3530532, -6.15546965, -2.59969421],
    [20.19334879, 20.08829657, -7.77023371, -21.23900361],
    [3.6099784, 13.81273842, -7.33082734, 1.13976051],
    [1.39849959, 6.01611686, -8.17928216, -10.05243078],
    [6.62841218, 8.5349038, -0.68728777, -6.57698286],
    [4.41993557, -2.33549325, -3.55681184, -14.12492204],
    [5.91283584, 11.82468148, -4.74599464, -5.81793792],
    [9.86837492, 28.41112199, -15.75363663, -23.77087632],
    [-3.34679152, 8.32705657, -23.16584027, -28.55832548],
    [6.60695424, 13.62665593, -13.13681748, -14.65493574],
    [12.17083426, 15.46848419, -5.10377601, -16.15614314],
    [19.5525709, 5.47367682, -11.28678722, -21.47738397],
    [14.23755811, 15.84170605, -16.63543265, -27.17113675],
    [20.19842771, 0.39614635, -24.06597556, -4.30602008],
    [17.30934004, 15.96123079, 2.2244595, -4.4779091],
    [12.25407557, 0.75645946, -3.48270365, -22.02973903],
    [7.6811488, -6.56575976, -5.43265598, -29.57499466],
    [16.55319629, 12.43843366, -19.2816324, -7.81510986],
    [4.4823932, 15.54617773, -11.26261732, -12.11752828],
    [17.45110058, 2.65822453, 5.34345221, -10.6916372],
    [0.88684841, 34.35616358, -19.95672452, -6.43595185],
    [0.63651552, 2.56246592, -8.90066952, -8.64849996],
    [2.64690715, 14.80659859, -18.21526889, -14.56501242],
    [1.73355176, 6.38091322, -6.85954974, -7.30401162],
    [13.01848837, 9.78463647, -7.28317682, -31.06160092],
    [8.52557501, 12.50291773, -5.03141506, -4.42999221],
    [1.86796955, 23.50428816, -17.65812717, -4.72199553],
    [2.61403391, 3.380633, -8.40220668, -13.21053888],
    [23.01464296, 33.52761557, -7.80505973, -28.43973305],
    [20.53116843, 23.42974028, -25.58164416, -9.80294036],
    [11.62227127, 8.07896416, -8.61519903, -21.18743346],
    [16.32443046, 16.64267921, -15.22826183, -9.56775284],
    [17.06417791, 13.54030207, -7.90733122, -34.47669187],
    [3.51776725, 6.02233748, -5.40791095, -4.83617388],
    [20.15455684, 16.19399739, 6.09167016, -12.12838473],
    [10.78471489, 10.53813329, -3.5760074, -7.45363648],
    [8.23124383, 5.19529104, 2.24486886, -3.55988963],
    [13.28786923, 14.53427974, -20.01163999, -10.83783583]]

pdis = PDIS("data.csv", 0, 200000)
pdis.calculate_pi_b()
# print("Student_T_Test", pdis.estimate_J_theta_candidate([-4.52376687, 2.71429282, -9.23526859, 8.41318765]))
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
