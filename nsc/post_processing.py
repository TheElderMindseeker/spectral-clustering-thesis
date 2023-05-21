import csv
from sys import maxsize
from collections import defaultdict, namedtuple

import numpy as np


Tree = namedtuple('Tree', ('x', 'y', 'height', 'crown_radius', 'points'))


def merge(tree_a, tree_b):
    all_points = np.row_stack((tree_a.points, tree_b.points))
    x_min, y_min = np.min(all_points[:, :2], axis=0)
    x_max, y_max = np.max(all_points[:, :2], axis=0)
    highest_idx = np.argmax(all_points[:, 2])
    x_tree_top = all_points[highest_idx, 0]
    y_tree_top = all_points[highest_idx, 1]
    height = all_points[highest_idx, 2]
    crown_radius = (x_max - x_min + y_max - y_min) / 4
    return Tree(x_tree_top, y_tree_top, height, crown_radius, all_points)


def post_process(detection_in, segmentation_in, avg_diameter, detection_out, segmentation_out, filtered_out, append):
    points = defaultdict(list)
    csv_reader = csv.reader(segmentation_in)
    next(csv_reader)  # Skip header.
    for row in csv_reader:
        points[int(row[-1])].append([float(v) for v in row[:3]])

    trees = []
    csv_reader = csv.reader(detection_in)
    next(csv_reader)  # Skip header.
    for row in csv_reader:
        tree_id = int(row[0])
        tree_points = np.array(points[tree_id])
        tree_x = float(row[1])
        tree_y = float(row[2])
        height = float(row[3])
        crown = float(row[4])
        trees.append(Tree(tree_x, tree_y, height, crown, tree_points))

    merged_trees = []
    for tree in trees:
        if tree.height >= 10:
            merged_trees.append(tree)
        else:
            tree_idx = 0
            while tree_idx < len(merged_trees):
                m_tree = merged_trees[tree_idx]
                if m_tree.height >= 10:
                    tree_idx += 1
                    continue
                tree_pos = np.array((tree.x, tree.y))
                m_tree_pos = np.array((m_tree.x, m_tree.y))
                if np.linalg.norm(tree_pos - m_tree_pos) < avg_diameter:
                    break
                tree_idx += 1
            if tree_idx < len(merged_trees):
                combined_tree = merge(tree, merged_trees[tree_idx])
                merged_trees.pop(tree_idx)
                merged_trees.append(combined_tree)

    trees = merged_trees
    good_trees = []
    bad_trees = []

    for tree in trees:
        x_min, y_min = np.min(tree.points[:, :2], axis=0)
        x_max, y_max = np.max(tree.points[:, :2], axis=0)
        ew_crown = x_max - x_min
        sn_crown = y_max - y_min
        crown_diff = np.abs(ew_crown - sn_crown)
        crown_mean = (ew_crown + sn_crown) / 2
        if tree.crown_radius > tree.height / 2:
            bad_trees.append(tree)
        elif crown_diff > crown_mean:
            bad_trees.append(tree)
        else:
            good_trees.append(tree)

    mode = 'a' if append > 0 else 'w'
    trees_csv = open(detection_out, mode)
    points_csv = open(segmentation_out, mode)
    trees_writer = csv.writer(trees_csv)
    points_writer = csv.writer(points_csv)
    if append == 0:
        trees_writer.writerow(('TreeID', 'TreeTopX', 'TreeTopY', 'Height', 'CrownRadius'))
        points_writer.writerow(('X', 'Y', 'Z', 'TreeID'))

    for tree_id, tree in enumerate(good_trees):
        trees_writer.writerow((append + tree_id, *tree[:-1]))
        for row in range(tree.points.shape[0]):
            points_writer.writerow((*tree.points[row], append + tree_id))

    trees_csv.close()
    points_csv.close()

    csv_writer = csv.writer(filtered_out)
    for tree in bad_trees:
        for row in range(tree.points.shape[0]):
            csv_writer.writerow(tree.points[row])

    return len(good_trees)


def combine_with_adaptive_height(detection_parts, segmentation_parts, height_thresholds, detection_out, segmentation_out):
    height_thresholds = list(height_thresholds)
    height_thresholds.insert(0, -maxsize - 1)
    height_thresholds.append(maxsize)

    trees_csv = open(detection_out, 'w')
    points_csv = open(segmentation_out, 'w')
    trees_writer = csv.writer(trees_csv)
    points_writer = csv.writer(points_csv)
    trees_writer.writerow(('TreeID', 'TreeTopX', 'TreeTopY', 'Height', 'CrownRadius'))
    points_writer.writerow(('X', 'Y', 'Z', 'TreeID'))

    id_counter = 0
    for i in range(1, len(height_thresholds)):
        points = defaultdict(list)
        segmentation_file = open(segmentation_parts[i - 1])
        csv_reader = csv.reader(segmentation_file)
        next(csv_reader)  # Skip header.
        for row in csv_reader:
            points[int(row[-1])].append([float(v) for v in row[:3]])

        trees = []
        detection_file = open(detection_parts[i - 1])
        csv_reader = csv.reader(detection_file)
        next(csv_reader)  # Skip header.
        for row in csv_reader:
            tree_id = int(row[0])
            tree_points = np.array(points[tree_id])
            tree_x = float(row[1])
            tree_y = float(row[2])
            height = float(row[3])
            crown = float(row[4])
            if height_thresholds[i - 1] <= height < height_thresholds[i]:
                trees.append(Tree(tree_x, tree_y, height, crown, tree_points))

        for tree in trees:
            trees_writer.writerow((id_counter, *tree[:-1]))
            for row in range(tree.points.shape[0]):
                points_writer.writerow((*tree.points[row], id_counter))
            id_counter += 1

    trees_csv.close()
    points_csv.close()
