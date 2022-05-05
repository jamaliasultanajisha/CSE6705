from json import tool
import random
import numpy
import operator
import itertools
import math
from deap import creator, base, tools, algorithms, benchmarks
from deap.benchmarks import movingpeaks
import SwarmPackagePy as sp
from SwarmPackagePy import intelligence
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import timeit


rnd = random.Random()
rnd.seed(128)

scenario = movingpeaks.SCENARIO_3

scenario["uniform_height"] = 0
scenario["uniform_width"] = 0
scenario["npeaks"] = 10
scenario["period"] = 1000

# sp.pso(50, tf.easom_function, -10, 10, 2, 20, w=0.5, c1=1, c2=1) 

NDIM = 2 #2 #5
BOUNDS = [scenario["min_coord"], scenario["max_coord"]]

mpb = movingpeaks.MovingPeaks(dim=NDIM, random=rnd, **scenario)
mpb.changePeaks()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, best=None, bestfit=creator.FitnessMax)
creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)


class PSO():
    def generate(pclass, dim, pmin, pmax, smin, smax):
        # part = pclass(random.uniform(pmin, pmax) for _ in range(dim))
        part = pclass(random.uniform(0, 0) for _ in range(dim)) 
        part.speed = [random.uniform(smin, smax) for _ in range(dim)]
        return part

    def updateParticle(part, best, chi, c, dir=0):
        ### user input here: input angle for particle postion (not speed) + forward cos0, - backward cos180 cos 30, cos 60
        part_dir = (math.cos(dir) for _ in range(len(part)))

        ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
        ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
    
        ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
        ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
        a = map(operator.sub,
                        map(operator.mul,
                                        itertools.repeat(chi),
                                        map(operator.add, ce1_p, ce2_g)),
                        map(operator.mul,
                                        itertools.repeat(1 - chi),
                                        part.speed))
        
        a = map(operator.mul, a, part_dir)

        part.speed = list(map(operator.add, part.speed, a))
        part[:] = list(map(operator.add, part, part.speed))
        # print(part[:])

    def convertQuantum(swarm, rcloud, centre, dist):
        dim = len(swarm[0])
        for part in swarm:
            position = [random.gauss(0, 1) for _ in range(dim)]
            dist = math.sqrt(sum(x**2 for x in position))

            if dist == "gaussian":
                u = abs(random.gauss(0, 1.0/3.0))
                part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

            elif dist == "uvd":
                u = random.random()
                part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

            elif dist == "nuvd":
                u = abs(random.gauss(0, 1.0/3.0))
                part[:] = [(rcloud * x * u / dist) + c for x, c in zip(position, centre)]

            del part.fitness.values
            del part.bestfit.values
            part.best = None

        return swarm

toolbox = base.Toolbox()
toolbox.register("particle", PSO.generate, creator.Particle, dim=NDIM, pmin=BOUNDS[0], pmax=BOUNDS[1], 
                 smin=-(BOUNDS[1] - BOUNDS[0])/2.0, smax=(BOUNDS[1] - BOUNDS[0])/2.0)
toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
toolbox.register("update", PSO.updateParticle, chi=0.729843788, c=2.05)
toolbox.register("convert", PSO.convertQuantum, dist="nuvd")
toolbox.register("evaluate", mpb)

def main(verbose=True):
    NSWARMS = 5
    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 1    # 0.5 times the move severity

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "nswarm", "evals", "error", "offline_error", "avg", "max"

    # Generate the initial population
    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]

    ans = input("Enter direction? (Y/N): ")
    if ans == "Y" or ans == "y":
        user_dir = input("Enter the direction: ")
    else:
        user_dir = 0

    # Evaluate each particle
    for swarm in population:
        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)

            # Update swarm's attractors personal best and global best
            if not part.best or part.fitness > part.bestfit:
                part.best = toolbox.clone(part[:])          # Get the position
                part.bestfit.values = part.fitness.values   # Get the fitness
            if not swarm.best or part.fitness > swarm.bestfit:
                swarm.best = toolbox.clone(part[:])         # Get the position
                swarm.bestfit.values = part.fitness.values  # Get the fitness

    record = stats.compile(itertools.chain(*population))
    logbook.record(gen=0, evals=mpb.nevals, nswarm=len(population),
                   error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

    if verbose:
        print(logbook.stream)

    generation = 1
    while mpb.nevals < 5e5 and mpb.currentError() > 0 and mpb.offlineError() > 0:
    # while generation < 5000:
        # Check for convergence
        rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population)**(1.0/NDIM))

        not_converged = 0
        worst_swarm_idx = None
        worst_swarm = None
        for i, swarm in enumerate(population):
            # Compute the diameter of the swarm
            for p1, p2 in itertools.combinations(swarm, 2):
                d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1, p2)))
                if d > 2*rexcl:
                    not_converged += 1
                    # Search for the worst swarm according to its global best
                    if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                        worst_swarm_idx = i
                        worst_swarm = swarm
                    break

        # If all swarms have converged, add a swarm
        if not_converged == 0:
            if worst_swarm_idx != None: ## added
                population.pop(worst_swarm_idx)  ## added 
            # If too little swarms roaming 
            # # if not_converged < NSWARMS: 
            population.append(toolbox.swarm(n=NPARTICLES)) ## here
            ## added
            reinit_swarms = set()
            for s1, s2 in itertools.combinations(range(len(population)), 2):
                # Swarms must have a best and not already be set to reinitialize
                if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
                    dist = 0
                    for x1, x2 in zip(population[s1].best, population[s2].best):
                        dist += (x1 - x2)**2.
                    dist = math.sqrt(dist)
                    if dist < rexcl:
                        if population[s1].bestfit <= population[s2].bestfit:
                            reinit_swarms.add(s1)
                        else:
                            reinit_swarms.add(s2)

            # Reinitialize and evaluate swarms
            for s in reinit_swarms:
                population[s] = toolbox.swarm(n=NPARTICLES)
                for part in population[s]:
                    part.fitness.values = toolbox.evaluate(part)

                    # Update swarm's attractors personal best and global best
                    if not part.best or part.fitness > part.bestfit:
                        part.best = toolbox.clone(part[:])
                        part.bestfit.values = part.fitness.values
                    if not population[s].best or part.fitness > population[s].bestfit:
                        population[s].best = toolbox.clone(part[:])
                        population[s].bestfit.values = part.fitness.values
        
        # elif len(population) < NSWARMS: 
        #     population.append(toolbox.swarm(n=NPARTICLES)) ## here

        # If too many swarms are roaming, remove the worst swarm
        elif not_converged > NEXCESS:
            population.pop(worst_swarm_idx)

        # Update and evaluate the swarm
        for swarm in population:
            # Check for change
            if swarm.best and toolbox.evaluate(swarm.best) != swarm.bestfit.values:
                # Convert particles to quantum particles
                swarm[:] = toolbox.convert(swarm, rcloud=RCLOUD, centre=swarm.best)
                swarm.best = None
                del swarm.bestfit.values

            for part in swarm:
                # Not necessary to update if it is a new swarm
                # or a swarm just converted to quantum
                if swarm.best and part.best:
                    if generation == 1:
                        toolbox.update(part, swarm.best, dir = int(user_dir))
                    toolbox.update(part, swarm.best, dir=0)
                part.fitness.values = toolbox.evaluate(part)

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = toolbox.clone(part[:])
                    part.bestfit.values = part.fitness.values
                if not swarm.best or part.fitness > swarm.bestfit:
                    swarm.best = toolbox.clone(part[:])
                    swarm.bestfit.values = part.fitness.values

        record = stats.compile(itertools.chain(*population))
        logbook.record(gen=generation, evals=mpb.nevals, nswarm=len(population),
                       error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

        if verbose:
            print(logbook.stream)

        # Apply exclusion
        # reinit_swarms = set()
        # for s1, s2 in itertools.combinations(range(len(population)), 2):
        #     # Swarms must have a best and not already be set to reinitialize
        #     if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
        #         dist = 0
        #         for x1, x2 in zip(population[s1].best, population[s2].best):
        #             dist += (x1 - x2)**2.
        #         dist = math.sqrt(dist)
        #         if dist < rexcl:
        #             if population[s1].bestfit <= population[s2].bestfit:
        #                 reinit_swarms.add(s1)
        #             else:
        #                 reinit_swarms.add(s2)

        # # Reinitialize and evaluate swarms
        # for s in reinit_swarms:
        #     population[s] = toolbox.swarm(n=NPARTICLES)
        #     for part in population[s]:
        #         part.fitness.values = toolbox.evaluate(part)

        #         # Update swarm's attractors personal best and global best
        #         if not part.best or part.fitness > part.bestfit:
        #             part.best = toolbox.clone(part[:])
        #             part.bestfit.values = part.fitness.values
        #         if not population[s].best or part.fitness > population[s].bestfit:
        #             population[s].best = toolbox.clone(part[:])
        #             population[s].bestfit.values = part.fitness.values
        generation += 1

    # animation(numpy.array((population)), tf.easom_function, int(BOUNDS[0]), int(BOUNDS[1]))


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)  