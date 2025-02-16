import numpy as np
import geopandas as gpd
import pytest
from cellseg_models_pytorch.utils.vectorize import inst2gdf, sem2gdf, gaussian_smooth


def test_inst2gdf_basic():
    inst_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    type_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    class_dict = {1: 'type1', 2: 'type2', 3: 'type3'}
    
    gdf = inst2gdf(inst_map, type_map, class_dict=class_dict)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert 'id' in gdf.columns
    assert 'class_name' in gdf.columns
    assert 'geometry' in gdf.columns

def test_inst2gdf_with_smoothing():
    inst_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    type_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    class_dict = {1: 'type1', 2: 'type2', 3: 'type3'}
    
    gdf = inst2gdf(inst_map, type_map, class_dict=class_dict, smooth_func=gaussian_smooth)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert 'id' in gdf.columns
    assert 'class_name' in gdf.columns
    assert 'geometry' in gdf.columns

def test_sem2gdf_basic():
    sem_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    class_dict = {1: 'type1', 2: 'type2', 3: 'type3'}
    
    gdf = sem2gdf(sem_map, class_dict=class_dict)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert 'id' in gdf.columns
    assert 'class_name' in gdf.columns
    assert 'geometry' in gdf.columns

def test_sem2gdf_with_smoothing():
    sem_map = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    class_dict = {1: 'type1', 2: 'type2', 3: 'type3'}
    
    gdf = sem2gdf(sem_map, class_dict=class_dict, smooth_func=gaussian_smooth)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert 'id' in gdf.columns
    assert 'class_name' in gdf.columns
    assert 'geometry' in gdf.columns
