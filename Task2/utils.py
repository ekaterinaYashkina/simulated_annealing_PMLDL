import math
from geopy.distance import lonlat, distance


"""
Calculates the distance between the two nodes using euclidean formula


neighbours - list of tuples, nodes coordinates
first, second - indices of elements in neighbours list, between which the distance should be computed

"""
def euclidean_dist(first, second, neighbours):
    first_c = neighbours[first]
    second_c = neighbours[second]

    return math.sqrt((first_c[0] - second_c[0]) ** 2 + (first_c[1] - second_c[1]) ** 2)


"""
Calculates the distance between the two nodes with lat lon coordinates. This is done with 
geopy lib.


neighbours - list of tuples, nodes coordinates
first, second - indices of elements in neighbours list, between which the distance should be computed

"""

def long_lat_dist(first, second, neighbours):
    first_c = neighbours[first]
    second_c = neighbours[second]


    return distance(lonlat(*first_c), lonlat(*second_c)).kilometers



"""

Calculates the distance among the provided path.

Dots - list of elements that construct the path. Indices of coors
coors - list of tuples, nodes coordinates
dist - which formula to use for distance calculation (now available - 'lonlat' and 'euclidean')

"""

def calculate_path(dots, coors, dist = 'lonlat'):


    if dist == 'lonlat':
        dist_f = long_lat_dist
    elif dist == 'euclidean':
        dist_f = euclidean_dist
    else:
        raise ValueError("Provided distance function is not exist. Please, use lonlat or euclidean")

    dist_v = 0
    for i in range(len(dots)-1):

        dist_v+=dist_f(dots[i], dots[i+1], coors)

    dist_v+=dist_f(dots[len(dots)-1], dots[0], coors)

    return dist_v