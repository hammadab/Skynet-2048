from random import randint


class PuzzleModel:
    # generate a puzzle with two blocks initialized as '2' or '4'
    def __init__(self):
        self.space = [[0 for col in range(4)] for row in range(4)]  # generate an empty puzzle
        y1 = randint(0, 3)
        x1 = randint(0, 3)
        self.space[y1][x1] = 2 * randint(1, 2)  # initialize one block
        while True:
            y2 = randint(0, 3)
            x2 = randint(0, 3)
            if (y1 != y2) or (x1 != x2):
                self.space[y2][x2] = 2 * randint(1, 2)  # initialize second block
                break

    def left(self):
        illegal = True
        # sort
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(0, 3):
                    if self.space[y][x] == 0:
                        if self.space[y][x + 1] > 0:
                            illegal = False
                            self.space[y][x] = self.space[y][x + 1]
                            self.space[y][x + 1] = 0
        # add
        for y in range(0, 4):
            for x in range(0, 3):
                if self.space[y][x] == self.space[y][x + 1]:
                    illegal = False
                    self.space[y][x] += self.space[y][x + 1]
                    self.space[y][x + 1] = 0
        if illegal:
            return False
        # sort
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(0, 3):
                    if self.space[y][x] == 0:
                        self.space[y][x] = self.space[y][x + 1]
                        self.space[y][x + 1] = 0
        if (0 in self.space[0]) or (0 in self.space[1]) or (0 in self.space[2]) or (0 in self.space[3]):
            while True:
                y = randint(0, 3)
                x = randint(0, 3)
                if self.space[y][x] == 0:
                    self.space[y][x] = 2 * randint(1, 2)  # initialize another block
                    break
        return True

    def right(self):
        illegal = True
        # sort
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(3, 0, -1):
                    if self.space[y][x] == 0:
                        if self.space[y][x - 1] > 0:
                            illegal = False
                            self.space[y][x] = self.space[y][x - 1]
                            self.space[y][x - 1] = 0
        # add
        for y in range(0, 4):
            for x in range(3, 0, -1):
                if self.space[y][x] == self.space[y][x - 1]:
                    illegal = False
                    self.space[y][x] += self.space[y][x - 1]
                    self.space[y][x - 1] = 0
        if illegal:
            return False
        # sort
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(3, 0, -1):
                    if self.space[y][x] == 0:
                        self.space[y][x] = self.space[y][x - 1]
                        self.space[y][x - 1] = 0
        if (0 in self.space[0]) or (0 in self.space[1]) or (0 in self.space[2]) or (0 in self.space[3]):
            while True:
                y = randint(0, 3)
                x = randint(0, 3)
                if self.space[y][x] == 0:
                    self.space[y][x] = 2 * randint(1, 2)  # initialize another block
                    break
        return True

    def up(self):
        illegal = True
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(0, 3):
                    if self.space[y][x] == 0:
                        if self.space[y + 1][x] > 0:
                            illegal = False
                            self.space[y][x] = self.space[y + 1][x]
                            self.space[y + 1][x] = 0
        # add
        for x in range(0, 4):
            for y in range(0, 3):
                if self.space[y][x] == self.space[y + 1][x]:
                    illegal = False
                    self.space[y][x] += self.space[y + 1][x]
                    self.space[y + 1][x] = 0
        if illegal:
            return False
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(0, 3):
                    if self.space[y][x] == 0:
                        self.space[y][x] = self.space[y + 1][x]
                        self.space[y + 1][x] = 0
        if (0 in self.space[0]) or (0 in self.space[1]) or (0 in self.space[2]) or (0 in self.space[3]):
            while True:
                y = randint(0, 3)
                x = randint(0, 3)
                if self.space[y][x] == 0:
                    self.space[y][x] = 2 * randint(1, 2)  # initialize another block
                    break
        return True

    def down(self):
        illegal = True
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(3, 0, -1):
                    if self.space[y][x] == 0:
                        if self.space[y - 1][x] > 0:
                            illegal = False
                            self.space[y][x] = self.space[y - 1][x]
                            self.space[y - 1][x] = 0
        # add
        for x in range(0, 4):
            for y in range(3, 0, -1):
                if self.space[y][x] == self.space[y - 1][x]:
                    illegal = False
                    self.space[y][x] += self.space[y - 1][x]
                    self.space[y - 1][x] = 0
        if illegal:
            return False
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(3, 0, -1):
                    if self.space[y][x] == 0:
                        self.space[y][x] = self.space[y - 1][x]
                        self.space[y - 1][x] = 0
        if (0 in self.space[0]) or (0 in self.space[1]) or (0 in self.space[2]) or (0 in self.space[3]):
            while True:
                y = randint(0, 3)
                x = randint(0, 3)
                if self.space[y][x] == 0:
                    self.space[y][x] = 2 * randint(1, 2)  # initialize another block
                    break
        return True

    def get_space(self):
        spaceList = []
        for y in range(0, 4):
            for x in range(0, 4):
                spaceList.append(self.space[y][x])
        spaceTemp = self.space.copy()
        matches = 0
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(0, 3):
                    if spaceTemp[y][x] == 0:
                        if spaceTemp[y][x + 1] > 0:
                            spaceTemp[y][x] = spaceTemp[y][x + 1]
                            spaceTemp[y][x + 1] = 0
        # count
        for y in range(0, 4):
            for x in range(0, 3):
                if spaceTemp[y][x] != 0:
                    if spaceTemp[y][x] == spaceTemp[y][x + 1]:
                        matches += 1
        spaceList.append(matches)
        spaceTemp = self.space.copy()
        matches = 0
        # sort
        for _ in range(0, 4):
            for y in range(0, 4):
                for x in range(3, 0, -1):
                    if spaceTemp[y][x] == 0:
                        if spaceTemp[y][x - 1] > 0:
                            spaceTemp[y][x] = spaceTemp[y][x - 1]
                            spaceTemp[y][x - 1] = 0
        # count
        for y in range(0, 4):
            for x in range(3, 0, -1):
                if spaceTemp[y][x] != 0:
                    if spaceTemp[y][x] == spaceTemp[y][x - 1]:
                        matches += 1
        spaceList.append(matches)
        spaceTemp = self.space.copy()
        matches = 0
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(0, 3):
                    if spaceTemp[y][x] == 0:
                        if spaceTemp[y + 1][x] > 0:
                            spaceTemp[y][x] = spaceTemp[y + 1][x]
                            spaceTemp[y + 1][x] = 0
        # count
        for x in range(0, 4):
            for y in range(0, 3):
                if spaceTemp[y][x] != 0:
                    if spaceTemp[y][x] == spaceTemp[y + 1][x]:
                        matches += 1
        spaceList.append(matches)
        spaceTemp = self.space.copy()
        matches = 0
        # sort
        for _ in range(0, 4):
            for x in range(0, 4):
                for y in range(3, 0, -1):
                    if spaceTemp[y][x] == 0:
                        if spaceTemp[y - 1][x] > 0:
                            spaceTemp[y][x] = spaceTemp[y - 1][x]
                            spaceTemp[y - 1][x] = 0
        # count
        for x in range(0, 4):
            for y in range(3, 0, -1):
                if spaceTemp[y][x] != 0:
                    if spaceTemp[y][x] == spaceTemp[y - 1][x]:
                        matches += 1
        spaceList.append(matches)
        return spaceList

    def game_won(self):
        if (2048 in self.space[0]) or (2048 in self.space[1]) or (2048 in self.space[2]) or (2048 in self.space[3]):
            return True
        return False

    def game_over(self):
        if (0 not in self.space[0]) and (0 not in self.space[1]) and (0 not in self.space[2]) and (
                0 not in self.space[3]):
            return True
        return False

    def score(self):
        return max(max(self.space[0]), max(self.space[1]), max(self.space[2]), max(self.space[3]))
