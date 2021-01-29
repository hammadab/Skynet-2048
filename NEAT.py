from PuzzleModel import PuzzleModel
import neat
import time
import os
import math
import pickle

ge = None
best_genome = None
winner = None
best_score = 0
gen = 0
prev_size = 0
# functions
def eval_genomes(genomes, config):
    global best_genome, best_score, gen, prev_size, ge
    ge = genomes
    # ix = [-1, 0, 1, 2, 3, 4]
    for le, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pm = PuzzleModel()  # initialize a puzzle
        # genome.fitness = pm.score()
        steps = 0
        poss = True
        while (not pm.game_won()) and (not pm.game_over()) and poss:
            steps += 1
            x = pm.get_space()
            # output = net.activate([temp / max(x) for temp in x])
            output = net.activate(x)
            i = output.index(max(output))
            # if i in ix:
            #     print(i)
            #     ix.remove(i)
            if i == 0:
                poss = pm.left()
            elif i == 1:
                poss = pm.right()
            elif i == 2:
                poss = pm.up()
            else:
                poss = pm.down()
        genome.fitness = pm.score()
        if genome.fitness > best_score:
            best_genome = genome
            best_score = genome.fitness
        if pm.game_won():
            genome.fitness = 2048
        elif pm.game_over():
            genome.fitness = (genome.fitness - steps) / 2
        else:
            genome.fitness = genome.fitness - steps
    gen += 1
    # if gen % 50 == 0:
    if prev_size != le:
        print("At gen " + str(gen) + " population size = " + str(len(genomes)))
        prev_size = le


def run(config_file):
    global best_genome, winner
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(1000))

    winner = p.run(eval_genomes, 50)

    pickle_out = open("winner.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()

    pickle_out = open("best_genome.pickle", "wb")
    pickle.dump(best_genome, pickle_out)
    pickle_out.close()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    winner_nnet = neat.nn.FeedForwardNetwork.create(winner, config)
    pm = PuzzleModel()
    poss = True
    while (not pm.game_won()) and (not pm.game_over()) and poss:
        x = pm.get_space()
        # output = winner_nnet.activate([temp / max(x) for temp in x])
        output = winner_nnet.activate(x)
        i = output.index(max(output))
        if i == 0:
            poss = pm.left()
        elif i == 1:
            poss = pm.right()
        elif i == 2:
            poss = pm.up()
        else:
            poss = pm.down()
        print(pm.space[0])
        print(pm.space[1])
        print(pm.space[2])
        print(pm.space[3])
        print("score = " + str(pm.score()))

    # Display my genome.
    print('\nMy genome:\n{!s}'.format(best_genome))
    best_nnet = neat.nn.FeedForwardNetwork.create(best_genome, config)
    pm = PuzzleModel()
    poss = True
    while (not pm.game_won()) and (not pm.game_over()) and poss:
        x = pm.get_space()
        # output = best_nnet.activate([temp / max(x) for temp in x])
        output = best_nnet.activate(x)
        i = output.index(max(output))
        if i == 0:
            poss = pm.left()
        elif i == 1:
            poss = pm.right()
        elif i == 2:
            poss = pm.up()
        else:
            poss = pm.down()
        print(pm.space[0])
        print(pm.space[1])
        print(pm.space[2])
        print(pm.space[3])
        print("score = " + str(pm.score()))


# main
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    # config_path = "/content/drive/My Drive/CS 464-1 Introduction to Machine Learning/skynet-2048/config-feedforward.txt"
    a = time.time()
    run(config_path)
    print("time taken = " + str(time.time() - a))
    input("Press enter to exit")  # to keep the output window open
