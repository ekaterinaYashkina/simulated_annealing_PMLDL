import random
import utils
import numpy as np

class SimulatedAnnealing():

    '''
    Neighbours - list of tuples with lon lat coordinates [(lon, lat)]
    cool_rate - the value on which Temperature should be decreased
    T - temperature
    min_T - minimum T to achieve to stop the iterations
    iters - number of iteration to run the algorithm
    annealing_shedule - whether T needs to be increased with some shedule (not every iteration),
                        if True, then ann_iter value should be provided
    ann_iter - provided if annealing_shedule is True, how many iterations to perform to change T
    '''

    def __init__(self, neighbours, cool_rate = 0.95, T = 1000, min_T = 1e-8, iters = 1000, annealing_shedule = False, ann_iter = None,
                 dist_func = 'lonlat', no_change = 1000):
        self.neighbours = neighbours
        self.cool_rate = cool_rate
        self.T = T
        self.min_T = min_T
        self.iters = iters
        self.cur_sol = [] # current sequence of nodes for TSP problem
        self.cur_dist = 0 # current distance for the cities sequence
        self.best_sol = [] # the best sequence with min dist, appeared during SA. This value might not be
                            #equal to the cur_sol, due to the randomness nature of SA
        self.best_dist = 0 # the min dist for all SA iterations
        self.fit_hist = []
        self.dist_func = dist_func

        # If T needs to be decreased with some schedule, the period number should be provided
        if annealing_shedule:
            assert ann_iter is not None
        self.annealing_shedule = annealing_shedule
        self.ann_iter = ann_iter
        self.counter = 0
        self.no_change = no_change


    '''
    
    Run the SA algorithm.
    The path is update if it is better than the current solution, otherwise randomly according to
    U(0, 1) < alpha, where U - uniform distribution, alpha - rate of previous and current scores
    
    params:
    
    update_policy - which algorithm to use for new path generation
        1) 2opt - choose the path snippet and reverse it (so A->B->C becomes C->B->A). It is used in
        most TSP problems and proven to be optimal
        
        2) simp_swap - just interchange two points (so A->B->C->D becomes D->B->C->A if we change A and D)
    
    '''


    def solve(self, verbose = True, update_policy = '2opt'): #update policy either 2opt or simp_swap

        # Initializition of the first path
        cur_sol = random.choice(list(np.arange(len(self.neighbours))))

        self.cur_sol.append(cur_sol)
        neighbours = list(np.arange(len(self.neighbours)))
        neighbours.remove(cur_sol)
        left_n = len(neighbours)
        for i in range(left_n):
            next_node = min(neighbours, key=lambda x: utils.long_lat_dist(cur_sol, x, self.neighbours))  # nearest neighbour
            neighbours.remove(next_node)
            self.cur_sol.append(next_node)
            cur_sol = next_node
        self.cur_dist = utils.calculate_path(self.cur_sol, self.neighbours, self.dist_func)
        self.best_sol = self.cur_sol
        self.best_dist = self.cur_dist
        self.fit_hist.append((self.cur_sol, self.cur_dist))

        cur_iter = 0

        # start the SA process
        while self.T > self.min_T and cur_iter<self.iters:
            new_path = self.cur_sol.copy()
            prev = self.cur_dist

            # generate new path with 2opt policy, the min path should be of length 2, so we keep the two more slots for the end
            if update_policy == '2opt':
                i2 = int(random.choice(np.arange(2, len(self.neighbours) - 1)))
                i1 = int(random.choice(np.arange(len(self.neighbours) - i2)))
                i2 = i1+i2
                new_path[i1:i2] = reversed(new_path[i1:i2])

            # generate new path with simply swapping two cities
            elif update_policy == 'simp_swap':
                i1 = random.choice(list(np.arange(len(self.neighbours))))
                i2 = random.choice(list(np.arange(len(self.neighbours))))
                saved = new_path[i1]
                new_path[i1] = new_path[i2]
                new_path[i2] = saved
            else:
                raise ValueError("The update policy can be either 2opt or simp_swap")

            # get the score for the new path and apply decision SA rule
            new_dist = utils.calculate_path(new_path, self.neighbours, self.dist_func)
            if new_dist < self.cur_dist:
                self.cur_dist, self.cur_sol = new_dist, new_path
                if new_dist < self.best_dist:
                    self.best_dist, self.best_sol = new_dist, new_path
            else:
                alpha = np.exp(-(new_dist-self.cur_dist)/self.T)
                if np.random.uniform(0, 1) <= alpha:
                    self.cur_dist, self.cur_sol = new_dist, new_path



            # print the information
            if verbose and cur_iter%10 == 0:
                print("Iteration {}/{} current distance - {} T - {}".format(cur_iter, self.iters, self.cur_dist, self.T))
            # change the temperature
            if self.annealing_shedule:
                if cur_iter%self.ann_iter == 0:
                    self.T*=self.cool_rate
            else:
                self.T *= self.cool_rate
            cur_iter+=1

            self.fit_hist.append((self.cur_sol, self.cur_dist))
            if prev == self.cur_dist:
                self.counter+=1
                if self.counter >= self.no_change:
                    break
            else:
                self.counter = 0



