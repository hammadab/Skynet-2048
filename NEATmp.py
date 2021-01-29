from PuzzleModel import PuzzleModel
import neat
import os
import multiprocessing
import time
import pickle

best_score = 0
gen = 0
prev_size = 0


# functions
def puzzle(genome):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(os.path.dirname(__file__), "config-feedforward.txt"))
    net = neat.nn.FeedForwardNetwork.create(genome[1], config)
    pm = PuzzleModel()  # initialize a puzzle
    steps = 0
    poss = True
    while (not pm.game_won()) and (not pm.game_over()) and poss:
        steps += 1
        x = pm.get_space()
        # output = net.activate([temp / max(x) for temp in x])
        output = net.activate(x)
        i = output.index(max(output))
        if i == 0:
            poss = pm.left()
        elif i == 1:
            poss = pm.right()
        elif i == 2:
            poss = pm.up()
        else:
            poss = pm.down()
    genome[1].fitness = pm.score()
    if pm.game_won():
        genome[1].fitness = 2048
    elif pm.game_over():
        genome[1].fitness = (genome[1].fitness - steps) / 2
    else:
        genome[1].fitness = genome[1].fitness - steps
    return genome[1].fitness


def eval_genomes(genomes, _):
    global best_genome, best_score, gen, prev_size
    gen += 1
    if prev_size != len(genomes):
        prev_size = len(genomes)
        print("At gen " + str(gen) + " population size = " + str(prev_size))
    if __name__ == '__main__':
        with multiprocessing.Pool() as p:
            fit = p.map(puzzle, genomes)
        p.close()
        for ix in range(0, len(fit)):
            genomes[ix][1].fitness = fit[ix]
        if max(fit) > best_score:
            best_score = max(fit)
            best_genome = genomes[fit.index(best_score)][1]


# main
if __name__ == '__main__':
    global best_genome
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_file = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    # config_file = "/content/drive/My Drive/CS 464-1 Introduction to Machine Learning/skynet-2048/config-feedforward.txt"
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(1000))

    a = time.time()
    winner = p.run(eval_genomes, 50)
    a = time.time() - a

    pickle_out = open("mp winner.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()

    pickle_out = open("mp best_genome.pickle", "wb")
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
    print("time taken = " + str(a))
    # input("Press enter to exit")  # to keep the output window open
