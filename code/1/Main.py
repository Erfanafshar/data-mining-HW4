import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

K = 4
epochs = 15
dataSetNum = 1
plot = True

ds1 = pd.read_csv("Dataset1.csv").to_numpy()
ds2 = pd.read_csv("Dataset2.csv").to_numpy()

if dataSetNum == 1:
    ds = ds1
else:
    ds = ds2


def plot_1():
    plt.scatter(ds[:, 0], ds[:, 1], color="red")
    plt.show()


def get_color(index):
    if index == 0:
        return "red"
    if index == 1:
        return "blue"
    if index == 2:
        return "yellow"
    if index == 3:
        return "orange"


def plot_2(points_center):
    for i in range(len(ds)):
        plt.scatter(ds[i][0], ds[i][1], color=get_color(points_center[i]))
    plt.show()


def initialize_centers():
    center_indexes = random.sample(range(0, len(ds)), K)
    centers = []
    for idx in center_indexes:
        centers.append(ds[idx])
    return centers


def calculate_distances(point, centers):
    distances = []
    for center_point in centers:
        distances.append((center_point[0] - point[0]) ** 2 + (center_point[1] - point[1]) ** 2)
    return distances


def set_point_center(distances, points_center):
    min_idx = distances.index(min(distances))
    points_center.append(min_idx)
    return points_center


def clear_centers(centers):
    center_num = []
    for _ in centers:
        center_num.append(0)
    centers = []
    for _ in center_num:
        centers.append(np.array([0.0, 0.0]))
    return centers, center_num


def k_means():
    centers = initialize_centers()
    for iter in range(epochs):
        points_center = []
        for point in ds:
            distances = calculate_distances(point, centers)
            points_center = set_point_center(distances, points_center)

        centers, center_num = clear_centers(centers)

        for i in range(len(ds)):
            centers[points_center[i]][0] += ds[i][0]
            centers[points_center[i]][1] += ds[i][1]
            center_num[points_center[i]] += 1

        for i in range(len(centers)):
            centers[i][0] /= center_num[i]
            centers[i][1] /= center_num[i]

    return centers, points_center


def initialize_mean_error():
    mean_error = []
    mean_error_num = []
    for i in range(len(centers)):
        mean_error.append(0)
        mean_error_num.append(0)
    return mean_error, mean_error_num


def calculate_cluster_error(centers, points_center):
    mean_error, mean_error_num = initialize_mean_error()
    for itr in range(len(ds)):
        distance = (centers[points_center[itr]][0] - ds[itr][0]) ** 2
        distance += (centers[points_center[itr]][1] - ds[itr][1]) ** 2
        mean_error[points_center[itr]] += distance
        mean_error_num[points_center[itr]] += 1

    for i in range(len(mean_error)):
        mean_error[i] /= mean_error_num[i]
    return mean_error


def print_mean_error():
    for itm in mean_error:
        print(itm)


def calc_print_clustering_error(mean_error):
    average_error = np.mean(mean_error)
    print()
    print(average_error)


centers, points_center = k_means()
mean_error = calculate_cluster_error(centers, points_center)
print_mean_error()
calc_print_clustering_error(mean_error)

if plot:
    plot_2(points_center)

