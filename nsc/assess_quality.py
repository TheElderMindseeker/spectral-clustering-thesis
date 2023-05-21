import csv
from os.path import exists
from collections import namedtuple

import fiona
import numpy as np
from shapely import Point
from shapely.geometry import shape


QualityParams = namedtuple(
    'QualityParams',
    ('detected', 'matched', 'commission', 'ommission', 'h_mean', 'v_mean'),
)


def assess(aoi_shapefile, tree_data_csv, ref_shapefile, similarity, match_writer, nomatch_writer):
    with fiona.open(aoi_shapefile) as shape_file:
        geometries = []
        for record in shape_file:
            geometries.append(shape(record['geometry']))
    area_of_interest = geometries[0]
    test_trees = []
    with open(tree_data_csv) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header.
        for row in csv_reader:
            x_tree, y_tree, height = float(row[1]), float(row[2]), float(row[3])
            test_trees.append((x_tree, y_tree, height))
    ref_trees = []
    with fiona.open(ref_shapefile) as shape_file:
        for record in shape_file:
            ref_trees.append(
                (*record.geometry['coordinates'], record.properties['HRef'])
            )
    matches = match_trees(test_trees, ref_trees)

    for t_idx, r_idx in matches.items():
        t_tree = test_trees[t_idx]
        r_tree = ref_trees[r_idx]
        match_writer.writerow((
            similarity,
            r_tree[-1],
            t_tree[-1],
            *r_tree[:2],
            *t_tree[:2],
        ))

    for r_idx in range(len(ref_trees)):
        if r_idx not in matches.values():
            nomatch_writer.writerow((similarity, ref_trees[r_idx][-1]))

    quality_params = assess_quality(
        test_trees,
        ref_trees,
        matches,
        area_of_interest,
        rate=True,
    )
    print(f'Detection rate: {100 * quality_params.detected:.2f}%')
    print(f'Matching rate: {100 * quality_params.matched:.2f}%')
    return quality_params


def assess_quality(test, reference, matches, aoi, rate=False):
    """Assess quality of individual tree detection

    Args:
        test: Array of tuples (x, y, height) of extracted trees.
        reference: Array of tuples (x, y, height) of reference trees.
        matches: Dict mapping indices of trees in test set to indices of trees in
            reference set.
        aoi: Area of interest. Test points outside that have not been matched are not
            counted in detected and commission.
        rate: If True, parameters are returned normalized by number of reference trees.

    Returns:
        Named tuple QualityParams where:

        0. detected -- total number of detected trees.
        1. matched -- total number of matched trees.
        2. commission -- total number of test trees that could not be matched.
        3. ommission -- total number of reference trees that could not be matched.
        4. h_mean -- mean of horizontal modulus of matching vectors.
        5. v_mean -- mean of tree height differences.

    """
    n_test = 0
    for t_idx, tree in enumerate(test):
        tree_pos = Point(tree[0], tree[1])
        if aoi.contains(tree_pos) or t_idx in matches.keys():
            n_test += 1
    n_ref = len(reference)
    n_match = len(matches)
    detected = n_test / n_ref if rate else n_test
    matched = n_match / n_ref if rate else n_match
    commission = (n_test - n_match) / n_ref if rate else n_test - n_match
    ommission = (n_ref - n_match) / n_ref if rate else n_ref - n_match
    h_values = []
    v_values = []
    for t_idx, r_idx in matches.items():
        t_tree = np.array(test[t_idx])
        r_tree = np.array(reference[r_idx])
        h_values.append(np.linalg.norm(t_tree[:2] - r_tree[:2]))
        v_values.append(np.abs(t_tree[2] - r_tree[2]))
    h_mean = np.array(h_values).mean()
    v_mean = np.array(v_values).mean()
    return QualityParams(detected, matched, commission, ommission, h_mean, v_mean)


def match_trees(
    test,
    reference,
    height_limits=(10, 15, 25),
    dh_bounds=(3, 3, 4, 5),
    dd_bounds=(3, 4, 5, 5),
    candidate_jump=2.5,
):
    """Matches trees from test set to reference set

    Eysn, Lothar, et al. "A benchmark of lidar-based single tree detection methods
    using heterogeneous forest data from the alpine space." Forests 6.5 (2015):
    1721-1747.

    Args:
        test: Array of tuples (x, y, height) of extracted trees.
        reference: Array of tuples (x, y, height) of reference trees.
        height_limits: Height values for determining bounds for dH and d2D.
        dh_bounds: Array of values of maximum dH for matching. Must be one element
            longer than height_limits.
        dd_bounds: Array of values of maximum d2D for matching. Must be one element
            longer than height_limits.
        candidate_jump: Value in meters that spatially limits possible candidate jumps.

    Returns:
        Dict mapping indices of trees in test set to indices of trees in reference set.
            Not all trees are necessarily matched.

    """
    if not isinstance(test, np.ndarray):
        test = np.array(test)
    if not isinstance(reference, np.ndarray):
        reference = np.array(reference)
    sorted_test = np.flip(np.argsort(test[:, 2]))
    matches = {}
    for t_idx in sorted_test:
        t_position = test[t_idx, :2]
        t_height = test[t_idx, 2]
        b_idx = 0
        while b_idx < len(height_limits) and t_height > height_limits[b_idx]:
            b_idx += 1
        h_bound = dh_bounds[b_idx]
        d_bound = dd_bounds[b_idx]
        candidates = []
        for r_idx, r_tree in enumerate(reference):
            dH = np.abs(t_height - r_tree[2])
            d2D = np.linalg.norm(t_position - r_tree[:2])
            if dH < h_bound and d2D < d_bound and r_idx not in matches.values():
                candidates.append(r_idx)
        candidates = sorted(
            candidates,
            key=lambda idx: np.linalg.norm(t_position - reference[idx][:2]),
        )
        try:
            best_idx = candidates[0]
        except IndexError:
            continue
        best_tree = reference[best_idx]
        best_dH = np.abs(t_height - best_tree[2])
        initial_d2D = np.linalg.norm(best_tree[:2] - t_position)
        for candidate in candidates[1:]:
            c_tree = reference[candidate]
            c_tree_dH = np.abs(t_height - c_tree[2])
            if c_tree_dH < best_dH:
                c_tree_d2D = np.linalg.norm(c_tree[:2] - t_position)
                if np.abs(c_tree_d2D - initial_d2D) < candidate_jump:
                    best_idx = candidate
                    best_tree = c_tree
                    best_dH = c_tree_dH
        best_d2D = np.linalg.norm(best_tree[:2] - t_position)
        for tc_idx in sorted_test:
            check_dH = np.abs(best_tree[2] - test[tc_idx, 2])
            check_dD = np.linalg.norm(best_tree[:2] - test[tc_idx, :2])
            if check_dD < best_d2D and check_dH < best_dH:
                break
        else:
            matches[t_idx] = best_idx
    return matches
