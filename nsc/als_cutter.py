import fiona
import laspy
import rasterio
import numpy as np
import rasterio.transform
from shapely.geometry import shape


def cut_als(aoi_shapefile, als_file, vegetation, dtm_file, margin, output_las_file):
    with fiona.open(aoi_shapefile) as shape_file:
        geometries = []
        for record in shape_file:
            geometries.append(shape(record['geometry']))

    area_of_interest = geometries[0]
    x_min, y_min, x_max, y_max = area_of_interest.bounds

    als_data = laspy.read(als_file)
    if isinstance(vegetation, int):
        is_vegetation = als_data.classification == vegetation
    else:
        is_vegetation = als_data.classification == vegetation[0]
        for veg in vegetation[1:]:
            is_vegetation = np.logical_or(
                is_vegetation,
                als_data.classification == veg
            )
    in_area_of_interest = np.logical_and(
        np.logical_and(
            als_data.x > x_min - margin,
            als_data.x < x_max + margin
        ),
        np.logical_and(
            als_data.y > y_min - margin,
            als_data.y < y_max + margin
        )
    )
    vegetation_in_aoi = np.logical_and(is_vegetation, in_area_of_interest)
    als_data.points = als_data.points[vegetation_in_aoi]

    with rasterio.open(dtm_file) as dtm_file:
        dtm_transform = dtm_file.transform
        elevation_band = dtm_file.read(1)

    rows, cols = rasterio.transform.rowcol(
        dtm_transform,
        als_data.xyz[:, 0],
        als_data.xyz[:, 1],
    )
    ground_level = elevation_band[rows, cols]
    als_data.z = als_data.z - ground_level

    smol_als = laspy.create(
        point_format=als_data.header.point_format,
        file_version=als_data.header.version
    )
    smol_als.points = als_data.points
    smol_als.write(output_las_file)
