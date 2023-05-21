import csv
import time

import laspy
import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from .original import VoxelNystromSC


def gaussian(point_a, point_b, weight_a, weight_b, scale_xy, scale_z):
    xy_distance = np.linalg.norm(point_a[:2] - point_b[:2])
    z_distance = point_a[2] - point_b[2]
    scaled_distance = (xy_distance / scale_xy)**2 + (z_distance / scale_z)**2
    return np.exp(-weight_a * weight_b * scaled_distance)


def cosine(high_point, low_point, weight_high, weight_low):
    if high_point[2] < low_point[2]:
        return cosine(low_point, high_point, weight_low, weight_high)
    origin_point = np.copy(high_point)
    origin_point[2] = min(0.0, low_point[2] - 1.0)
    vector_a = high_point - origin_point
    vector_b = low_point - origin_point
    return (
        np.dot(vector_a, vector_b)
        / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    )


def quadratic(point_a, point_b, weight_a, weight_b, scale_xy, scale_z, alpha=1):
    xy_distance = np.linalg.norm(point_a[:2] - point_b[:2])
    z_distance = point_a[2] - point_b[2]
    scaled_distance = (xy_distance / scale_xy)**2 + (z_distance / scale_z)**2
    return np.power(1 + weight_a * weight_b * scaled_distance / (2 * alpha), -alpha)


def arctangent(point_a, point_b, weight_a, weight_b, scale_xy, scale_z):
    xy_distance = np.linalg.norm(point_a[:2] - point_b[:2])
    z_distance = point_a[2] - point_b[2]
    scaled_distance = (xy_distance / scale_xy)**2 + (z_distance / scale_z)**2
    return 1 - 2 * np.arctan(weight_a * weight_b * scaled_distance) / np.pi


def detect_and_segment(input_cloud, k_neighbors, sigma_xy_sqrt, sigma_z_sqrt, eps, max_trees, detection_csv, segmentation_csv, similarity='gaussian', bandwidth=None):
    if isinstance(input_cloud, str):
        lidar_data = laspy.read(input_cloud)
        point_data = lidar_data.xyz
    else:
        csv_reader = csv.reader(input_cloud)
        points = []
        for row in csv_reader:
            points.append([float(v) for v in row])
        point_data = np.array(points)

    x_min, y_min = np.min(point_data[:, :2], axis=0)
    x_max, y_max = np.max(point_data[:, :2], axis=0)
    plot_area = (x_max - x_min) * (y_max - y_min)
    point_density = point_data.shape[0] / plot_area
    quantile = 1 / plot_area
    if bandwidth is None:
        bandwidth = estimate_bandwidth(point_data, quantile=quantile, n_jobs=-1)
        if point_density > 80:
            bandwidth /= 2

    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    grouping_start = time.time()
    mean_shift.fit(point_data)
    grouping_time = time.time() - grouping_start
    group_labels = mean_shift.labels_
    group_centers = mean_shift.cluster_centers_
    group_weights = np.array([
        np.where(group_labels == group_idx)[0].size
        for group_idx in range(group_centers.shape[0])
    ])
    num_groups = group_centers.shape[0]

    knn_start = time.time()
    # In the original work, points are viewed in XY-plane for this.
    knn_adjacency = kneighbors_graph(
        group_centers[:, :2],
        k_neighbors,
        include_self=True,
        n_jobs=-1,
    )
    knn_time = time.time() - knn_start

    group_ids = np.array(range(num_groups))
    node_knns = np.reshape(np.nonzero(knn_adjacency)[1], (num_groups, k_neighbors))
    similarity_knn = np.zeros(node_knns.shape)
    for s_grp_idx in range(similarity_knn.shape[0]):
        s_weight = group_weights[s_grp_idx]
        for column in range(node_knns.shape[1]):
            d_grp_idx = node_knns[s_grp_idx, column]
            d_weight = group_weights[d_grp_idx]
            if similarity == 'gaussian':
                similarity_knn[s_grp_idx, column] = gaussian(
                    group_centers[s_grp_idx],
                    group_centers[d_grp_idx],
                    s_weight,
                    d_weight,
                    sigma_xy_sqrt,
                    sigma_z_sqrt,
                )
            elif similarity == 'cosine':
                similarity_knn[s_grp_idx, column] = cosine(
                    group_centers[s_grp_idx],
                    group_centers[d_grp_idx],
                    s_weight,
                    d_weight,
                )
            elif similarity == 'cosine_squared':
                similarity_knn[s_grp_idx, column] = cosine(
                    group_centers[s_grp_idx],
                    group_centers[d_grp_idx],
                    s_weight,
                    d_weight,
                )**2
            elif similarity == 'quadratic':
                similarity_knn[s_grp_idx, column] = quadratic(
                    group_centers[s_grp_idx],
                    group_centers[d_grp_idx],
                    s_weight,
                    d_weight,
                    sigma_xy_sqrt,
                    sigma_z_sqrt,
                )
            elif similarity == 'arctangent':
                similarity_knn[s_grp_idx, column] = arctangent(
                    group_centers[s_grp_idx],
                    group_centers[d_grp_idx],
                    s_weight,
                    d_weight,
                    sigma_xy_sqrt,
                    sigma_z_sqrt,
                )
            else:
                raise ValueError(f'Unrecognized similarity option {similarity}')

    sort_order = np.flip(np.argsort(np.sum(similarity_knn, axis=1)))
    group_ids_sorted = group_ids[sort_order]

    landmarks = set()
    remaining = set()
    drop_counter = 0
    for group_idx in group_ids_sorted:
        if group_idx in remaining:
            continue
        if np.isclose(np.sum(similarity_knn[group_idx]), 1, atol=eps, rtol=0):
            remaining.add(group_idx)
            drop_counter += 1
            continue
        landmarks.add(group_idx)
        for k_neighbor in node_knns[group_idx]:
            if k_neighbor in landmarks:
                continue
            remaining.add(k_neighbor)

    landmarks = np.array(list(landmarks), dtype=np.int32)
    remaining = np.array(list(remaining), dtype=np.int32)
    similarity_landmarks = np.zeros((landmarks.size, landmarks.size))
    similarity_remaining = np.zeros((landmarks.size, remaining.size))

    for point_idx in landmarks:
        row = np.where(landmarks == point_idx)[0][0]
        for k in range(node_knns.shape[1]):
            neighbor_idx = node_knns[point_idx, k]
            if neighbor_idx in landmarks:
                column = np.where(landmarks == neighbor_idx)[0][0]
                similarity_landmarks[row, column] = similarity_knn[point_idx, k]
                similarity_landmarks[column, row] = similarity_knn[point_idx, k]
            else:
                column = np.where(remaining == neighbor_idx)[0][0]
                similarity_remaining[row, column] = similarity_knn[point_idx, k]

    matrix_is_pos_def = np.all(np.linalg.eigvals(similarity_landmarks) > 0)
    degrees = np.sum(similarity_knn, axis=1)
    norm_degrees = np.sqrt(degrees)
    d_norm_upper = norm_degrees[landmarks]
    d_norm_lower = norm_degrees[remaining]
    normed_landmarks = (
        np.eye(landmarks.size) -
        similarity_landmarks / np.outer(d_norm_upper, d_norm_upper)
    )
    normed_remaining = -(
        similarity_remaining / np.outer(d_norm_upper, d_norm_lower)
    )

    matrix_is_pos_def = np.all(np.linalg.eigvals(normed_landmarks) > 0)
    if matrix_is_pos_def:
        linalg_time = time.time()
        A_sqrt = sqrtm(normed_landmarks)
        if np.isclose(np.linalg.det(A_sqrt), 0, atol=eps**2):
            A_sqrt_inv = np.linalg.pinv(A_sqrt)
        else:
            A_sqrt_inv = np.linalg.inv(A_sqrt)
        matrix_S = (
            normed_landmarks
            + A_sqrt_inv @ normed_remaining @ normed_remaining.T @ A_sqrt_inv
        )
        eigenvalues, matrix_U = np.linalg.eig(matrix_S)
        Lambda_S = np.diag(eigenvalues)
        Lambda_S_sqrt_inv = np.linalg.inv(sqrtm(Lambda_S))
        stack_matrix = np.vstack((normed_landmarks, normed_remaining.T))
        eigenvectors = stack_matrix @ A_sqrt_inv @ matrix_U @ Lambda_S_sqrt_inv
        linalg_time = time.time() - linalg_time
    else:
        raise ValueError('Laplacian matrix block A is not positive definite')

    max_clusters = np.min((max_trees, eigenvalues.size - 1))
    half_max_clusters = max_clusters // 2
    eigen_ids_sorted = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[eigen_ids_sorted]
    eigengap = (
        eigenvalues_sorted[(half_max_clusters + 1):(max_clusters + 1)]
        - eigenvalues_sorted[half_max_clusters:max_clusters]
    )
    largest_gap_idx = np.argsort(eigengap)[-1]
    num_clusters = half_max_clusters + largest_gap_idx
    eigenvectors_first = eigenvectors[:, eigen_ids_sorted[:num_clusters]]
    eigenvectors_normed = normalize(eigenvectors_first)
    cluster_labels = KMeans(
        num_clusters,
        n_init='auto',
        random_state=42,
    ).fit_predict(eigenvectors_normed)

    tree_clusters = np.zeros((point_data.shape[0],), dtype=np.int32)
    for label_idx in range(cluster_labels.size):
        if label_idx < landmarks.size:
            group_idx = landmarks[label_idx]
        else:
            group_idx = remaining[label_idx - landmarks.size]
        tree_clusters[group_labels == group_idx] = cluster_labels[label_idx]

    clusters = np.sort(np.unique(tree_clusters))
    trees = []
    for cluster_idx in clusters:
        cluster_points = point_data[tree_clusters == cluster_idx]
        x_min, y_min = np.min(cluster_points[:, :2], axis=0)
        x_max, y_max = np.max(cluster_points[:, :2], axis=0)
        highest_idx = np.argmax(cluster_points[:, 2])
        x_tree_top = cluster_points[highest_idx, 0]
        y_tree_top = cluster_points[highest_idx, 1]
        height = cluster_points[highest_idx, 2]
        crown_radius = (x_max - x_min + y_max - y_min) / 4
        trees.append((cluster_idx, x_tree_top, y_tree_top, height, crown_radius))

    csv_writer = csv.writer(detection_csv)
    csv_writer.writerow(('TreeID', 'TreeTopX', 'TreeTopY', 'Height', 'CrownRadius'))
    csv_writer.writerows(trees)

    csv_writer = csv.writer(segmentation_csv)
    csv_writer.writerow(('X', 'Y', 'Z', 'TreeID'))
    for point_idx in range(tree_clusters.size):
        csv_writer.writerow((*point_data[point_idx], tree_clusters[point_idx]))

    return bandwidth


def use_original(input_cloud, max_trees, detection_csv, segmentation_csv, bandwidth):
    if isinstance(input_cloud, str):
        lidar_data = laspy.read(input_cloud)
        point_data = lidar_data.xyz
    else:
        csv_reader = csv.reader(input_cloud)
        points = []
        for row in csv_reader:
            points.append([float(v) for v in row])
        point_data = np.array(points)
    return VoxelNystromSC(
        point_data,
        max_trees,
        segmentation_csv,
        detection_csv,
        bandwidth,
    )
