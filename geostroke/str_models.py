"""Haversine comparator retained from the AUDEBERT cross-sectional notebook."""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def gravity_alloc(hospitals, counties, beta=2.0, max_km=150):
    allocation = np.zeros(len(counties))
    population = counties.total_pop.to_numpy(dtype=float)
    for hospital in hospitals.itertuples():
        distance = np.clip(haversine_km(hospital.latitude, hospital.longitude,
                           counties.centroid_lat.to_numpy(), counties.centroid_lon.to_numpy()), 1.0, None)
        include = distance <= max_km
        weights = np.zeros(len(counties))
        weights[include] = population[include] / distance[include]**beta
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights[np.argmin(distance)] = 1.0
        allocation += hospital.total_cases * weights
    return allocation
