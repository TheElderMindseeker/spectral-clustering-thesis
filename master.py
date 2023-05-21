import csv
from os.path import exists, join
from tempfile import TemporaryFile

import numpy as np

from nsc import *


DATASET_PATH = './NEWFOR/Benchmark_Data/'
WORKING_PATH = './newfor/'
TREE_MULTIPLIER = 1.5
MEAN_DIAMETER = 5.0

common_settings = {
    'vegetation': 4,
    'margin': 3.0,
    'k_neighbors': 1 + 40,  # Self + neighbors.
    'sigma_xy': np.sqrt(10),
    'sigma_z': np.sqrt(6 * 10),
    'eps': 1e-5,
}

saint_agnan_01 = join(DATASET_PATH, '01_Saint_Agnan/')
working_01 = join(WORKING_PATH, '01_Saint_Agnan/')
Saint_Agnan_01 = {
    'aoi': join(saint_agnan_01, '01_AoI.shp'),
    'ref': join(saint_agnan_01, '01_Ref.shp'),
    'als': join(saint_agnan_01, '01_ALS.las'),
    'dtm': join(saint_agnan_01, '01_DTM.tif'),
    'cut_als': join(working_01, '01_ALS_cut.las'),
    'detection': join(working_01, 'detection.csv'),
    'segmentation': join(working_01, 'segmentation.csv'),
    'max_trees': int(359 * TREE_MULTIPLIER),
    'plot_number': 1,
    'n_reference': 359,
    'forest_type': 'ML/M',
    'stem_density': 359,
}

cotolivier_02 = join(DATASET_PATH, '02_Cotolivier/')
working_02 = join(WORKING_PATH, '02_Cotolivier/')
Cotolivier_02 = {
    'vegetation': 1,
    'margin': 10.0,
    'aoi': join(cotolivier_02, '02_AoI.shp'),
    'ref': join(cotolivier_02, '02_Ref.shp'),
    'als': join(cotolivier_02, '02_ALS_v.las'),
    'dtm': join(cotolivier_02, '02_DTM.tif'),
    'cut_als': join(working_02, '02_ALS_cut.las'),
    'detection': join(working_02, 'detection.csv'),
    'segmentation': join(working_02, 'segmentation.csv'),
    'max_trees': int(106 * TREE_MULTIPLIER),
    'plot_number': 2,
    'n_reference': 106,
    'forest_type': 'ML/C',
    'stem_density': 843,
}

cotolivier_03 = join(DATASET_PATH, '03_Cotolivier/')
working_03 = join(WORKING_PATH, '03_Cotolivier/')
Cotolivier_03 = {
    'vegetation': 1,
    'aoi': join(cotolivier_03, '03_AoI.shp'),
    'ref': join(cotolivier_03, '03_Ref.shp'),
    'als': join(cotolivier_03, '03_ALS_v.las'),
    'dtm': join(cotolivier_03, '03_DTM.tif'),
    'cut_als': join(working_03, '03_ALS_cut.las'),
    'detection': join(working_03, 'detection.csv'),
    'segmentation': join(working_03, 'segmentation.csv'),
    'max_trees': int(49 * TREE_MULTIPLIER),
    'plot_number': 3,
    'n_reference': 49,
    'forest_type': 'SL/C',
    'stem_density': 390,
}

cotolivier_04 = join(DATASET_PATH, '04_Cotolivier/')
working_04 = join(WORKING_PATH, '04_Cotolivier/')
Cotolivier_04 = {
    'vegetation': 1,
    'aoi': join(cotolivier_04, '04_AoI.shp'),
    'ref': join(cotolivier_04, '04_Ref.shp'),
    'als': join(cotolivier_04, '04_ALS_v.las'),
    'dtm': join(cotolivier_04, '04_DTM.tif'),
    'cut_als': join(working_04, '04_ALS_cut.las'),
    'detection': join(working_04, 'detection.csv'),
    'segmentation': join(working_04, 'segmentation.csv'),
    'max_trees': int(22 * TREE_MULTIPLIER),
    'plot_number': 4,
    'n_reference': 22,
    'forest_type': 'ML/M',
    'stem_density': 175,
}

montafon_06 = join(DATASET_PATH, '06_Montafon/')
working_06 = join(WORKING_PATH, '06_Montafon/')
Montafon_06 = {
    'vegetation': 0,
    'aoi': join(montafon_06, '06_AoI.shp'),
    'ref': join(montafon_06, '06_Ref.shp'),
    'als': join(montafon_06, '06_ALS_v.las'),
    'dtm': join(montafon_06, '06_DTM.tif'),
    'cut_als': join(working_06, '06_ALS_cut.las'),
    'detection': join(working_06, 'detection.csv'),
    'segmentation': join(working_06, 'segmentation.csv'),
    'max_trees': int(107 * TREE_MULTIPLIER),
    'plot_number': 6,
    'n_reference': 107,
    'forest_type': 'ML/C',
    'stem_density': 400,
}

pellizzano_07 = join(DATASET_PATH, '07_Pellizzano/')
working_07 = join(WORKING_PATH, '07_Pellizzano/')
Pellizzano_07 = {
    'vegetation': [3, 4, 5],
    'aoi': join(pellizzano_07, '07_AoI.shp'),
    'ref': join(pellizzano_07, '07_Ref.shp'),
    'als': join(pellizzano_07, '07_ALS.las'),
    'dtm': join(pellizzano_07, '07_DTM.tif'),
    'cut_als': join(working_07, '07_ALS_cut.las'),
    'detection': join(working_07, 'detection.csv'),
    'segmentation': join(working_07, 'segmentation.csv'),
    'max_trees': int(49 * TREE_MULTIPLIER),
    'plot_number': 7,
    'n_reference': 49,
    'forest_type': 'SL/C',
    'stem_density': 374,
}

pellizzano_08 = join(DATASET_PATH, '08_Pellizzano/')
working_08 = join(WORKING_PATH, '08_Pellizzano/')
Pellizzano_08 = {
    'vegetation': [3, 4, 5],
    'aoi': join(pellizzano_08, '08_AoI.shp'),
    'ref': join(pellizzano_08, '08_Ref.shp'),
    'als': join(pellizzano_08, '08_ALS.las'),
    'dtm': join(pellizzano_08, '08_DTM.tif'),
    'cut_als': join(working_08, '08_ALS_cut.las'),
    'detection': join(working_08, 'detection.csv'),
    'segmentation': join(working_08, 'segmentation.csv'),
    'max_trees': int(235 * TREE_MULTIPLIER),
    'plot_number': 8,
    'n_reference': 235,
    'forest_type': 'ML/M',
    'stem_density': 1870,
}

asiago_09 = join(DATASET_PATH, '09_Asiago/')
working_09 = join(WORKING_PATH, '09_Asiago/')
Asiago_09 = {
    'vegetation': 1,
    'aoi': join(asiago_09, '09_AoI.shp'),
    'ref': join(asiago_09, '09_Ref.shp'),
    'als': join(asiago_09, '09_ALS_v.las'),
    'dtm': join(asiago_09, '09_DTM.tif'),
    'cut_als': join(working_09, '09_ALS_cut.las'),
    'detection': join(working_09, 'detection.csv'),
    'segmentation': join(working_09, 'segmentation.csv'),
    'max_trees': int(80 * TREE_MULTIPLIER),
    'plot_number': 9,
    'n_reference': 80,
    'forest_type': 'SL/C',
    'stem_density': 708,
}

asiago_10 = join(DATASET_PATH, '10_Asiago/')
working_10 = join(WORKING_PATH, '10_Asiago/')
Asiago_10 = {
    'vegetation': 1,
    'aoi': join(asiago_10, '10_AoI.shp'),
    'ref': join(asiago_10, '10_Ref.shp'),
    'als': join(asiago_10, '10_ALS_v.las'),
    'dtm': join(asiago_10, '10_DTM.tif'),
    'cut_als': join(working_10, '10_ALS_cut.las'),
    'detection': join(working_10, 'detection.csv'),
    'segmentation': join(working_10, 'segmentation.csv'),
    'max_trees': int(110 * TREE_MULTIPLIER),
    'plot_number': 10,
    'n_reference': 110,
    'forest_type': 'ML/M',
    'stem_density': 851,
}

asiago_11 = join(DATASET_PATH, '11_Asiago/')
working_11 = join(WORKING_PATH, '11_Asiago/')
Asiago_11 = {
    'vegetation': 1,
    'aoi': join(asiago_11, '11_AoI.shp'),
    'ref': join(asiago_11, '11_Ref.shp'),
    'als': join(asiago_11, '11_ALS_v.las'),
    'dtm': join(asiago_11, '11_DTM.tif'),
    'cut_als': join(working_11, '11_ALS_cut.las'),
    'detection': join(working_11, 'detection.csv'),
    'segmentation': join(working_11, 'segmentation.csv'),
    'max_trees': int(183 * TREE_MULTIPLIER),
    'plot_number': 11,
    'n_reference': 183,
    'forest_type': 'ML/M',
    'stem_density': 1344,
}

leskova_15 = join(DATASET_PATH, '15_Leskova/')
working_15 = join(WORKING_PATH, '15_Leskova/')
Leskova_15 = {
    'vegetation': [4, 5],
    'aoi': join(leskova_15, '15_AoI.shp'),
    'ref': join(leskova_15, '15_Ref.shp'),
    'als': join(leskova_15, '15_ALS_v.las'),
    'dtm': join(leskova_15, '15_DTM.tif'),
    'cut_als': join(working_15, '15_ALS_cut.las'),
    'detection': join(working_15, 'detection.csv'),
    'segmentation': join(working_15, 'segmentation.csv'),
    'max_trees': int(53 * TREE_MULTIPLIER),
    'plot_number': 15,
    'n_reference': 53,
    'forest_type': 'SL/M',
    'stem_density': 265,
}

leskova_16 = join(DATASET_PATH, '16_Leskova/')
working_16 = join(WORKING_PATH, '16_Leskova/')
Leskova_16 = {
    'vegetation': [4, 5],
    'aoi': join(leskova_16, '16_AoI.shp'),
    'ref': join(leskova_16, '16_Ref.shp'),
    'als': join(leskova_16, '16_ALS_v.las'),
    'dtm': join(leskova_16, '16_DTM.tif'),
    'cut_als': join(working_16, '16_ALS_cut.las'),
    'detection': join(working_16, 'detection.csv'),
    'segmentation': join(working_16, 'segmentation.csv'),
    'max_trees': int(37 * TREE_MULTIPLIER),
    'plot_number': 16,
    'n_reference': 37,
    'forest_type': 'SL/M',
    'stem_density': 185,
}

leskova_17 = join(DATASET_PATH, '17_Leskova/')
working_17 = join(WORKING_PATH, '17_Leskova/')
Leskova_17 = {
    'vegetation': [4, 5],
    'aoi': join(leskova_17, '17_AoI.shp'),
    'ref': join(leskova_17, '17_Ref.shp'),
    'als': join(leskova_17, '17_ALS_v.las'),
    'dtm': join(leskova_17, '17_DTM.tif'),
    'cut_als': join(working_17, '17_ALS_cut.las'),
    'detection': join(working_17, 'detection.csv'),
    'segmentation': join(working_17, 'segmentation.csv'),
    'max_trees': int(117 * TREE_MULTIPLIER),
    'plot_number': 17,
    'n_reference': 117,
    'forest_type': 'ML/M',
    'stem_density': 585,
}

leskova_18 = join(DATASET_PATH, '18_Leskova/')
working_18 = join(WORKING_PATH, '18_Leskova/')
Leskova_18 = {
    'vegetation': [4, 5],
    'aoi': join(leskova_18, '18_AoI.shp'),
    'ref': join(leskova_18, '18_Ref.shp'),
    'als': join(leskova_18, '18_ALS_v.las'),
    'dtm': join(leskova_18, '18_DTM.tif'),
    'cut_als': join(working_18, '18_ALS_cut.las'),
    'detection': join(working_18, 'detection.csv'),
    'segmentation': join(working_18, 'segmentation.csv'),
    'max_trees': int(92 * TREE_MULTIPLIER),
    'plot_number': 18,
    'n_reference': 92,
    'forest_type': 'ML/M',
    'stem_density': 460,
}

dataset = [
    Saint_Agnan_01,
    Cotolivier_02,
    Cotolivier_03,
    Cotolivier_04,
    Montafon_06,
    Pellizzano_07,
    Pellizzano_08,
    Asiago_09,
    Asiago_10,
    Asiago_11,
    Leskova_15,
    Leskova_16,
    Leskova_17,
    Leskova_18,
]


def detect_and_post_process(job, original, similarity, append=0, bandwidth=None):
    tmp_detection = TemporaryFile('w+')
    tmp_segmentation = TemporaryFile('w+')
    if original:
        bandwidth = use_original(
            job['cut_als'],
            job['max_trees'],
            tmp_detection,
            tmp_segmentation,
            bandwidth,
        )
    else:
        bandwidth = detect_and_segment(
            job['cut_als'],
            job['k_neighbors'],
            job['sigma_xy'],
            job['sigma_z'],
            job['eps'],
            job['max_trees'],
            tmp_detection,
            tmp_segmentation,
            similarity=similarity,
            bandwidth=bandwidth,
        )
    tmp_detection.seek(0)
    tmp_segmentation.seek(0)
    tmp_filtered = TemporaryFile('w+')
    num_good_trees = post_process(
        tmp_detection,
        tmp_segmentation,
        MEAN_DIAMETER,
        job['detection'],
        job['segmentation'],
        tmp_filtered,
        append,
    )
    tmp_detection.close()
    tmp_segmentation.close()
    return tmp_filtered, num_good_trees, bandwidth


def adaptive_experiment(targets):
    perf_path = 'results/performance.csv'
    match_path = 'results/match.csv'
    nomatch_path = 'results/nomatch.csv'
    result_append = exists(perf_path) and exists(match_path) and exists(nomatch_path)
    result_mode = 'a' if result_append else 'w'
    perf_file = open(perf_path, result_mode)
    match_file = open(match_path, result_mode)
    nomatch_file = open(nomatch_path, result_mode)
    perf_writer = csv.writer(perf_file)
    match_writer = csv.writer(match_file)
    nomatch_writer = csv.writer(nomatch_file)
    if not result_append:
        perf_writer.writerow((
            'SimilarityFunction',
            'PlotNumber',
            'TreesNumber',
            'ForestType',
            'StemDensity',
            'DetectionRate',
            'MatchingRate',
        ))
        match_writer.writerow((
            'SimilarityFunction',
            'ReferenceHeight',
            'EstimatedHeight',
            'ReferenceX',
            'ReferenceY',
            'EstimatedX',
            'EstimatedY',
        ))
        nomatch_writer.writerow((
            'SimilarityFunction',
            'TreeHeight',
        ))
    for target in targets:
        job = common_settings | target
        cut_als_path = job['cut_als']
        detection_out = job['detection']
        segmentation_out = job['segmentation']
        print('Processing', job['als'])
        cut_als(
            job['aoi'],
            job['als'],
            job['vegetation'],
            job['dtm'],
            job['margin'],
            job['cut_als'],
        )

        job['detection'] = 'newfor/detection_gaussian_part.csv'
        job['segmentation'] = 'newfor/segmentation_gaussian_part.csv'
        tmp_filtered, append, bandwidth = detect_and_post_process(job, False, 'gaussian')
        tmp_filtered.seek(0)
        job['cut_als'] = tmp_filtered
        final_filtered, _, _ = detect_and_post_process(job, False, 'gaussian', append, bandwidth)
        tmp_filtered.close()
        final_filtered.close()

        job['cut_als'] = cut_als_path
        job['detection'] = 'newfor/detection_cosine_part.csv'
        job['segmentation'] = 'newfor/segmentation_cosine_part.csv'
        tmp_filtered, append, bandwidth = detect_and_post_process(job, False, 'cosine')
        tmp_filtered.seek(0)
        job['cut_als'] = tmp_filtered
        final_filtered, _, _ = detect_and_post_process(job, False, 'cosine', append, bandwidth)
        tmp_filtered.close()
        final_filtered.close()

        combine_with_adaptive_height(
            (
                'newfor/detection_gaussian_part.csv',
                'newfor/detection_cosine_part.csv',
            ),
            (
                'newfor/segmentation_gaussian_part.csv',
                'newfor/segmentation_cosine_part.csv',
            ),
            (23,),
            detection_out,
            segmentation_out,
        )

        quality_params = assess(
            job['aoi'],
            detection_out,
            job['ref'],
            'adaptive',
            match_writer,
            nomatch_writer,
        )
        perf_writer.writerow((
            'adaptive',
            job['plot_number'],
            job['n_reference'],
            job['forest_type'],
            job['stem_density'],
            quality_params.detected,
            quality_params.matched,
        ))
    perf_file.close()
    match_file.close()
    nomatch_file.close()

def main(targets, similarity, original=False):
    perf_path = 'results/performance.csv'
    match_path = 'results/match.csv'
    nomatch_path = 'results/nomatch.csv'
    result_append = exists(perf_path) and exists(match_path) and exists(nomatch_path)
    result_mode = 'a' if result_append else 'w'
    perf_file = open(perf_path, result_mode)
    match_file = open(match_path, result_mode)
    nomatch_file = open(nomatch_path, result_mode)
    perf_writer = csv.writer(perf_file)
    match_writer = csv.writer(match_file)
    nomatch_writer = csv.writer(nomatch_file)
    if not result_append:
        perf_writer.writerow((
            'SimilarityFunction',
            'PlotNumber',
            'TreesNumber',
            'ForestType',
            'StemDensity',
            'DetectionRate',
            'MatchingRate',
        ))
        match_writer.writerow((
            'SimilarityFunction',
            'ReferenceHeight',
            'EstimatedHeight',
            'ReferenceX',
            'ReferenceY',
            'EstimatedX',
            'EstimatedY',
        ))
        nomatch_writer.writerow((
            'SimilarityFunction',
            'TreeHeight',
        ))
    for target in targets:
        job = common_settings | target
        print('Processing', job['als'])
        cut_als(
            job['aoi'],
            job['als'],
            job['vegetation'],
            job['dtm'],
            job['margin'],
            job['cut_als'],
        )
        tmp_filtered, append, bandwidth = detect_and_post_process(job, original, similarity)
        tmp_filtered.seek(0)
        job['cut_als'] = tmp_filtered
        final_filtered, _, _ = detect_and_post_process(job, original, similarity, append, bandwidth)
        tmp_filtered.close()
        final_filtered.close()
        quality_params = assess(
            job['aoi'],
            job['detection'],
            job['ref'],
            similarity,
            match_writer,
            nomatch_writer,
        )
        perf_writer.writerow((
            similarity,
            job['plot_number'],
            job['n_reference'],
            job['forest_type'],
            job['stem_density'],
            quality_params.detected,
            quality_params.matched,
        ))
    perf_file.close()
    match_file.close()
    nomatch_file.close()


if __name__ == '__main__':
    # main(dataset, 'gaussian')
    # main(dataset, 'cosine')
    # main(dataset, 'quadratic')
    # main(dataset, 'arctangent')
    adaptive_experiment(dataset)
