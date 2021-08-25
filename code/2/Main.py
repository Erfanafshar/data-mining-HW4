import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

K = 64
epochs = 10
plot = True

img = mpimg.imread("sample_img1.png")
img2d = img.transpose(2, 0, 1).reshape(3, -1).transpose()
ds = img2d


def initialize_centers():
    center_indexes = random.sample(range(0, len(ds)), K)
    centers = []
    for idx in center_indexes:
        centers.append(ds[idx])
    return centers


def calculate_distances(point, centers):
    distances = []
    for center_point in centers:
        distances.append((center_point[0] - point[0]) ** 2 +
                         (center_point[1] - point[1]) ** 2 +
                         (center_point[2] - point[2]) ** 2)
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
        centers.append(np.array([0.0, 0.0, 0.0]))
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
            centers[points_center[i]][2] += ds[i][2]
            center_num[points_center[i]] += 1

        for i in range(len(centers)):
            centers[i][0] /= center_num[i]
            centers[i][1] /= center_num[i]
            centers[i][2] /= center_num[i]

    return centers, points_center


def set_new_color(centers, points_center):
    for itr in range(len(points_center)):
        ds[itr][0] = centers[points_center[itr]][0]
        ds[itr][1] = centers[points_center[itr]][1]
        ds[itr][2] = centers[points_center[itr]][2]


def plot_new_image():
    img3d = ds.reshape(512, 512, 3)
    plt.imshow(img3d, interpolation='nearest')
    plt.show()


centers, points_center = k_means()
set_new_color(centers, points_center)

if plot:
    plot_new_image()

print("moz")
