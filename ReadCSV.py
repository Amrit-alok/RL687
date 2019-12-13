import csv


class ReadBehaviorFile:
    def __init__(self, file_name, start, end):
        #self.data_partition = data_partition
        with open(file_name, 'r') as csvfile:
            # creating a csv reader object
            csv_input = csv.reader(csvfile)
            self.data = list(csv_input)
        self.num_states = int(self.data[0][0])
        self.num_action = int(self.data[1][0])
        self.fourier_order = int(self.data[2][0])
        self.theta_b = self.data[3]
        self.num_episodes = int(self.data[4][0])
        self.policy = self.data[200005]
        self.trajectories = [[]]
        # for trajectory in self.data[5:200005]:
        for trajectory in self.data[start+5:end+5]:
            self.trajectories.append(list(map(float, trajectory)))

