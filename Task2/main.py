

import parser, sim_anneal, utils
from visulization.vis_data_proc import  animation, plot_change
import sys
import argparse

parser1 = argparse.ArgumentParser(allow_abbrev=False)
parser1.add_argument("--path_to_dataset", type = str, default="dataset/city.csv", help = "where the data is stored")
parser1.add_argument("--start_T", type = float, default=100, help = "starting temperature for SA")
parser1.add_argument("--iters", type = int, default=5000, help = "number of iterations")
parser1.add_argument("--ann_schedule", type = bool, default=True, help = "whether to use scheduling for T descrease")
parser1.add_argument("--ann_iters", type = int, default=10, help = "period for annealing scheduling")
parser1.add_argument("--gen_func", type = str, default='2opt', help = "which formula to use for new path generation: 2opt or simp_swap")
parser1.add_argument("--dist_func", type = str, default='lonlat', help = "which formula to use for distnce calculation: lonlat or euclidean")
parser1.add_argument("--plot", type = str, default="map", help="what to plot: map - the path change on the map, dist - distance change/per iteration")

arg = parser1.parse_args()

DS_PATH = arg.path_to_dataset


SORT_COLUMN = 'population'

COLUMNS_TO_KEEP = ['city', 'geo_lon', 'geo_lat', 'population']

DRAFT_CITY = 'address'

sorted_df = parser.process_csv(DS_PATH, SORT_COLUMN, COLUMNS_TO_KEEP, DRAFT_CITY)

names, coords = parser.convert_format(sorted_df, COLUMNS_TO_KEEP[1:3], COLUMNS_TO_KEEP[0])


sa = sim_anneal.SimulatedAnnealing(neighbours=coords, annealing_shedule=arg.ann_schedule, ann_iter=arg.ann_iters,
                                   T = arg.start_T, iters=arg.iters, dist_func=arg.dist_func)
sa.solve(verbose=True, update_policy=arg.gen_func)

hist = sa.fit_hist

if arg.plot == "map":

    animation(hist, coords, names)
elif arg.plot == "dist":
    plot_change(hist)



