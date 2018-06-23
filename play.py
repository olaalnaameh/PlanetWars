import argparse
import random
import time
import logging
from planetwars import PlanetWars
from planetwars.views import TextView
from ai.state import State
logging.basicConfig(filename='log.log',level=logging.INFO)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--collisions', action='store_true', required=False, default=False,
                        help="Should the ships collide among each other?")
    parser.add_argument('--rate', type=int, required=False, default=100,
                        help="Number of turns per second run by the game.")
    parser.add_argument('--map', type=str, required=False, default="map1",
                        help="The filename without extension for planets.")
    parser.add_argument('--quiet', action='store_true', required=False, default=True,
                        help="Suppress all output to the console.")

    parser.add_argument('--seed', type=int, required=False, default=0,
                        help="Initial rng seed, 0 = time-based")
    parser.add_argument('--p1num', type=int, required=False, default=1,
                        help="Planet number for player 1.")
    parser.add_argument('--p2num', type=int, required=False, default=1,
                        help="Planet number for player 2.")
    parser.add_argument('--nnum', type=int, required=False, default=10,
                        help="Number of neutral planets.")
    parser.add_argument('--genmaps', action='store_true', required=False, default=False,
                        help="Generate random maps.")
    parser.add_argument('--gennum', type=bool, required=False, default=False,
                        help="Generate player and neutral numbers at runtime with random seed.")
    parser.add_argument('--train', type=bool, required=False, default=False,
                        help="Do you want to train a bot")
    parser.add_argument('--games', type=int, required=False, default=1,
                        help="The number of games to play")

    arguments, remaining = parser.parse_known_args(argv)

    seed = 0
    if arguments.seed == 0:
      # use system seed and print the resulting random integer
      seed = random.randint(1, 2000000000)
    else:
      # use passed on seed
      seed = arguments.seed

    random.seed(seed)
    print ("seed=", seed)  #, "rnd1=", random.randint(1, 2000000000)            

    all_games_timing = time.time()
    winners = { 3 : 0, 0 : 0, 1 : 0, 2 : 0}
    logging.info('------------------------------- vs ' + str(remaining[:2][1]))
    for games in xrange(arguments.games):
        if arguments.genmaps:
            if arguments.gennum:
                player_starting_p = random.randint(1,2)
                arguments.p1num = player_starting_p
                arguments.p2num = player_starting_p
                arguments.nnum = random.randint(2, 5)
            print ("p1num=", arguments.p1num)
            print ("p2num=", arguments.p2num)
            print ("nnum=",  arguments.nnum)

        games_timing = time.time()
        if arguments.genmaps:
          state = State()
          state.random_setup(arguments.p1num, arguments.p2num, arguments.nnum)
          game = PlanetWars(remaining[:2], planets=state.planets, fleets=state.fleets, turns_per_second=arguments.rate, collisions=arguments.collisions, train=arguments.train)
        else:
          game = PlanetWars(remaining[:2], map_name=arguments.map, turns_per_second=arguments.rate, collisions=arguments.collisions, train=arguments.train)
      
        game.add_view(TextView(arguments.quiet))
        winner, _, _, _, _ = game.play()
        if winner == 1 or winner == 2 or winner == 3:
            winners[winner] += 1
        else:
            winners[0] += 1 # this is a tie
        logging.info(str(winners) + '     Time: ' + str(time.time() - games_timing))
        print ('---------------------------------------------------------')
        print ('Game time: ', time.time() - games_timing, ' seconds')
        print ('---------------------------------------------------------')   
         # before every game
        if arguments.train == True and (remaining[:2][0] == 'HeuristicNN' or remaining[:2][1] == 'HeuristicNN'):
            from planetwars.neuralnetwork import NNetwork
            NNetwork.save()

    print ('---------------------------------------------------------')
    print ('All games: ', time.time() - all_games_timing, ' seconds')
    print ('---------------------------------------------------------')
    print ('Ties        ' + str(remaining[:2][0]) + '        ' + remaining[:2][1])
    print (str(winners[0]) + '           ' + str(winners[1]) + '                  ' + str(winners[2]))
    print ('---------------------------------------------------------')
   

    

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
