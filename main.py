#! /usr/bin/env python

import numpy as np
from pysat.examples.fm import FM
from pysat.formula import WCNF
from pysat.solvers import Solver
from optparse import OptionParser
from datetime import datetime


def getStepsBound(jars):
    number_of_steps_bound = 1
    for jar in jars[0, :]:
        number_of_steps_bound *= (jar + 1)
    return number_of_steps_bound + 1


def getMultiIndex(step, volumes, dimensions):
    kek = np.concatenate([np.array([step]), volumes])
    return int(np.ravel_multi_index(kek, dimensions))


def makeJumps(model, dimensions, n):
    volumes = np.array([[0] * n])
    for i in model:
        if i > 0:
            index = np.array(np.unravel_index(i - 1, dimensions))
            volumes = np.append(volumes, [index[1:]], axis=0)
    return volumes[1:]


def printJars(jars):
    if len(jars) == 0:
        print("No solution")
    else:
        print("Number of steps:", len(jars) - 1)
        for i in range(len(jars) - 1):
            print(jars[i], end=" -> ")
        print(jars[-1])


def jump(jar_volumes, volumes):
    jumps = np.array([[0] * len(volumes)])
    for i in range(len(volumes)):
        for j in range(len(volumes)):
            if i == j:
                if volumes[i] != 0:
                    jumps = np.append(jumps, [volumes], axis=0)
                    jumps[-1, i] = 0
                if volumes[i] != jar_volumes[i]:
                    jumps = np.append(jumps, [volumes], axis=0)
                    jumps[-1, i] = jar_volumes[i]
            else:
                if volumes[j] != jar_volumes[j]:
                    vol = min(jar_volumes[j] - volumes[j], volumes[i])
                    jumps = np.append(jumps, [volumes], axis=0)
                    jumps[-1, i] -= vol
                    jumps[-1, j] += vol
    return jumps[1:]


def buildHardClauses(jars, goal, number_of_steps_bound, goalArr):
    number_of_states = number_of_steps_bound + 1
    for jar in jars[0, :]:
        number_of_states *= jar + 1

    dimensions = np.concatenate([np.array([number_of_steps_bound + 1]), jars[0, :] + 1])

    initial_clause = [[getMultiIndex(0, jars[1, :], dimensions) + 1]]
    step_clauses = [[] for i in range(number_of_states + 1)]
    single_state_clauses = []
    goal_clause = [[]]

    for state in range(number_of_states - 1, -1, -1):
        index = np.array(np.unravel_index(state, dimensions))
        current_step = index[0]
        volumes = index[1:]

        if current_step:
            step_clauses[state].append(-(state + 1))

        if len(goalArr) == 0 and goal in volumes or len(goalArr) != 0 and (goalArr == volumes).all():
            goal_clause[0].append(state + 1)

        if current_step < number_of_steps_bound:
            next_states = jump(jars[0, :], volumes)
            for i in range(len(next_states)):
                nxt = getMultiIndex(current_step + 1, next_states[i], dimensions)
                step_clauses[nxt].append(state + 1)

        for j in range(state + 1, number_of_states):
            index = np.unravel_index(j, dimensions)
            step = index[0]
            if step != current_step:
                break
            single_state_clauses.append([-(state + 1), -(j + 1)])

    return initial_clause + step_clauses + single_state_clauses + goal_clause, dimensions


def buildSoftClauses(jars, number_of_steps_bound):
    number_of_states = number_of_steps_bound + 1
    for jar in jars[0, :]:
        number_of_states *= jar + 1

    dimensions = np.concatenate([np.array([number_of_steps_bound + 1]), jars[0, :] + 1])
    soft_clauses = [[] for i in range(number_of_steps_bound + 1)]

    for state in range(number_of_states):
        index = np.array(np.unravel_index(state, dimensions))
        current_step = index[0]
        volumes = index[1:]

        soft_clauses[current_step].append(state + 1)

    return soft_clauses


def solve_cnf_linear_default(jars, goal, goalArr):
    number_of_steps_bound = getStepsBound(jars)

    for k in range(number_of_steps_bound):
        clauses, dimensions = buildHardClauses(jars, goal, k, goalArr)
        clauses = list(filter(bool, clauses))
        s = Solver(bootstrap_with=clauses)
        if s.solve():
            return makeJumps(s.get_model(), dimensions, len(jars[0]))
    return []


def solve_cnf_binary_default(jars, goal, goalArr):
    number_of_steps_bound = getStepsBound(jars)
    fl = False

    l = -1
    r = number_of_steps_bound
    while r - l > 1:
        m = (l + r) // 2

        clauses, dimensions = buildHardClauses(jars, goal, m, goalArr)
        clauses = list(filter(bool, clauses))
        s = Solver(bootstrap_with=clauses)

        if s.solve():
            r = m
            fl = True
        else:
            l = m

    if not fl:
        return []

    clauses, dimensions = buildHardClauses(jars, goal, r, goalArr)
    clauses = list(filter(bool, clauses))
    s = Solver(bootstrap_with=clauses)
    s.solve()
    return makeJumps(s.get_model(), dimensions, len(jars[0]))


def solve_cnf_linear_cardinality(jars, goal, goalArr):
    number_of_steps_bound = getStepsBound(jars)
    number_of_states = number_of_steps_bound ** 2

    clauses, dimensions = buildHardClauses(jars, goal, number_of_steps_bound, goalArr)
    clauses = list(filter(bool, clauses))
    full_clause = [i + 1 for i in range(number_of_states)]

    for k in range(number_of_steps_bound):
        s = Solver(name="Minicard", bootstrap_with=clauses)
        s.add_atmost(full_clause, k=k)
        if s.solve():
            return makeJumps(s.get_model(), dimensions, len(jars[0]))
    return []


def solve_cnf_binary_cardinality(jars, goal, goalArr):
    number_of_steps_bound = getStepsBound(jars)
    number_of_states = number_of_steps_bound ** 2

    clauses, dimensions = buildHardClauses(jars, goal, number_of_steps_bound, goalArr)
    clauses = list(filter(bool, clauses))
    full_clause = [i + 1 for i in range(number_of_states)]
    fl = False

    l = -1
    r = number_of_steps_bound
    while r - l > 1:
        m = (l + r) // 2
        s = Solver(name="Minicard", bootstrap_with=clauses)
        s.add_atmost(full_clause, k=m)

        if s.solve():
            r = m
            fl = True
        else:
            l = m

    if not fl:
        return []

    s = Solver(name="Minicard", bootstrap_with=clauses)
    s.add_atmost(full_clause, k=r)
    s.solve()
    return makeJumps(s.get_model(), dimensions, len(jars[0]))


def solve_wcnf(jars, goal, step, goalArr):
    number_of_steps_bound = getStepsBound(jars)

    for k in range(step, number_of_steps_bound // step * step + 1, step):
        number_of_states = number_of_steps_bound * (k + 1)
        hardClauses, dimensions = buildHardClauses(jars, goal, k, goalArr)
        hardClauses = list(filter(bool, hardClauses))
        softClauses = [[-(i + 1)] for i in range(number_of_states)]

        formula = WCNF()
        formula.extend(hardClauses)
        formula.extend(softClauses, [1] * len(softClauses))

        fm = FM(formula)
        if fm.compute():
            return makeJumps(fm.model, dimensions, len(jars[0]))
    return []


def solveJars(jars, goal, search, step, goalArr):
    if search == "linear":
        return solve_cnf_linear_default(jars, goal, goalArr)
    elif search == "linear_card":
        return solve_cnf_linear_cardinality(jars, goal, goalArr)
    elif search == "binary":
        return solve_cnf_binary_default(jars, goal, goalArr)
    elif search == "binary_card":
        return solve_cnf_binary_cardinality(jars, goal, goalArr)
    elif search == "weight":
        return solve_wcnf(jars, goal, step, goalArr)
    else:
        assert (False, "Something went wrong!")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--mode', help="mode to use:\n linear, linear_card, binary, binary_card, weight", default="linear", action="store", type="string", dest="mode")
    parser.add_option('-i', '--input-file', help="file with input data", default="", action="store", type="string", dest="file")
    parser.add_option('-s', '--step', help="step for cardinality/weight", default=5, action="store", type="int", dest="step")
    parser.add_option('-g', '--goal', help="goal in case you want to specify goal for each jar", default="", action="store", type="string", dest="goal")

    (options, args) = parser.parse_args()

    if len(options.file) == 0:
        n, goal, t = map(int, input().split())
        jars = np.zeros((2, n), dtype=int)
        jars[0, :] = list(map(int, input().split()))
        if t != 0:
            jars[1, :] = list(map(int, input().split()))
    else:
        file = open(options.file, "r")
        n, goal, t = map(int, file.readline().split())
        jars = np.zeros((2, n), dtype=int)
        jars[0, :] = list(map(int, file.readline().split()))
        if t != 0:
            jars[1, :] = list(map(int, file.readline().split()))
        file.close()
    start = datetime.now()
    goalArr = []
    if len(options.goal) != 0:
        goalArr = list(map(int, options.goal.split(',')))
    print("mode:", options.mode)
    printJars(solveJars(jars, goal, options.mode, options.step, goalArr))
    print(datetime.now() - start)
