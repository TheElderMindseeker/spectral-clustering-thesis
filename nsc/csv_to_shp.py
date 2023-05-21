import csv
from collections import namedtuple

import fiona


Tree = namedtuple('Tree', ('TreeID', 'TreeTopX', 'TreeTopY', 'Height', 'CrownRadius'))


def export_to_shp(ref_shp, input_csv, output_shp):
    with fiona.open(ref_shp) as ref_shp:
        driver = ref_shp.driver
        crs = ref_shp.crs
        schema = ref_shp.schema

    trees = []
    with open(input_csv) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header.
        for row in csv_reader:
            trees.append(Tree(*row))

    with fiona.open(output_shp, "w", driver=driver, crs=crs, schema=schema) as out_shp:
        for tree in trees:
            new_feature = {
                'id': int(tree.TreeID),
                'geometry': {
                    'type': 'Point',
                    'coordinates': (float(tree.TreeTopX), float(tree.TreeTopY))
                },
                'properties': {
                    'DBHRef': 0.0,
                    'HRef': float(tree.Height),
                    'VolRef': 0.0
                }
            }
            out_shp.write(fiona.Feature.from_dict(new_feature))
