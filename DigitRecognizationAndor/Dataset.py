import math

class Exemple:
    def __init__(self, entry, out):
        self.entry = entry
        self.out = out


def normalization(tab):
    u = 0
    for i in range(0, len(tab)):
        u = u + tab[j]
    u = u / tab.length


class Dataset:

    def normalize(self):
        for i in range(0, self.nb_entry):
            m = 0
            ecart = 0
            for j in range(0, self.nb_exemple):
                m = m + self.exemple_list[j].entry[i]
            m = m / self.nb_exemple
            for j in range(0, self.nb_exemple):
                ecart = ecart + (self.exemple_list[j].entry[i] - m) ** 2
            ecart = ecart / self.nb_exemple
            ecart = math.sqrt(ecart)
            for j in range(0, self.nb_exemple):
                self.exemple_list[j].entry[i] = (self.exemple_list[j].entry[i] - m) / ecart

    def __init__(self, file_name):
        file = open(file_name, "r")
        lines = file.read().splitlines()
        line_one = lines[0].split(' ')
        self.nb_exemple = int(line_one[0])
        self.nb_entry = int(line_one[1])
        self.nb_out = int(line_one[2])
        self.exemple_list = []
        i = 1
        while i < len(lines):
            entry = lines[i]
            out = lines[i + 1]
            i = i + 2
            entry = list(float(x) for x in entry.split())
            out = list(float(x) for x in out.split())
            self.exemple_list.append(Exemple(entry, out))

