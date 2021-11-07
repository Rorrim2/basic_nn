import numpy as np
import random

# def ordered_crossover(parentX, parentY):
#     start, end = 0, 0
#     childX = np.full(parentX.shape[0], -1)
#     childY = np.full(parentX.shape[0], -1)
#     while(np.abs(start - end) < 3):
#         start, end = sorted([random.randrange(parentX.shape[0]) for _ in range(2)])
#
#     childX[start:end] = parentX[start:end]
#     childY[start:end] = parentY[start:end]
#     i, j = 0, 0
#     cpX, cpY = 0, 0
#     tempX, tempY = 0, 0
#     if start != 0:
#         while(cpX != start or cpY != start):
#             tempX = parentX[i]
#             tempY = parentY[j]
#             if cpY != start and not np.any(childX[:] == tempY):
#                 childX[cpY] = tempY
#                 cpY += 1
#             if cpX != start and not np.any(childY[:] == tempX):
#                 childY[cpX] = tempX
#                 cpX += 1
#             if cpY != start:
#                 j += 1
#             if cpX != start:
#                 i += 1
#     cpY, cpX = end, end
#     while(cpX != parentY.shape[0] or cpY != parentY.shape[0]):
#         tempX = parentX[i]
#         tempY = parentY[j]
#         if cpY != parentY.shape[0] and not np.any(childX[:] == tempY):
#             childX[cpY] = tempY
#             cpY += 1
#         if cpX != parentY.shape[0] and not np.any(childY[:] == tempX):
#             childY[cpX] = tempX
#             cpX += 1
#         if cpY != parentY.shape[0]:
#             j += 1
#         if cpX != parentY.shape[0]:
#             i += 1
#     return childX, childY
#
# ordered_crossover(np.array([8,7,3,0,1,4,6,2,5]),
#                   np.array([3,8,2,7,6,0,1,5,4]))

visited = []

node1 = ((5,5), 'S', 1)
node2 = ((1,1), 'W', 1)
node3 = ((1,1), 'S', 1)

visited.append(node1)
visited.append(node2)

if node1 in visited:
    print("Test1 ok")

if node2 in visited:
    print("Test2 ok")

if node3 in visited:
    print("Test3 ok")


visited = []
    path = []
    check = []

    start = problem.getStartState()
    visited.append(start)
    reached = False
    dfsExternalFunction(visited, problem, start, path, reached)

    corrected_path = []
    for i in range(1, len(visited)-1):
        if problem.isGoalState(visited[i]):
            corrected_path.append(path[i - 1])
            break
        corrected_path.append(path[i-1])

    return corrected_path

def dfsExternalFunction(visited, problem, node, path, reached):
    from game import Directions
    if problem.isGoalState(node):
        reached = True
        return
    neighbours = problem.getSuccessors(node)
    for neighbour in neighbours:
        if neighbour[0] not in visited:
            entered = True
            path.append(neighbour[1])
            visited.append(neighbour[0])
            dfsExternalFunction(visited, problem, neighbour[0], path, reached)
            if not reached:
                path.append(Directions.REVERSE[neighbour[1]])
                visited.append(neighbour[0])

