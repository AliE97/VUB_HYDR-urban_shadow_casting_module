
from pysolar.solar import get_altitude, get_azimuth
from math import radians, sin, cos, tan, isnan
from datetime import timezone , timedelta
from numba import njit, prange
import xarray as xr
import pandas as pd
import numpy as np
import rasterio as rio
import os

from gdal import *


class Model:
    def __init__(
            self,
            dsm_map_path : str,
            dtm_map_path : str,
            land_use_map_path : str,
            output_path : str,
            simulation_date : str = "2025-06-21",
            transmissivity : float = 0.5,
            start_time : int = 1,
            end_time : int = 1,
            UTC: int = 0
            ):
            self.dsm_map_path = dsm_map_path
            self.dtm_map_path = dtm_map_path
            self.land_use_map_path = land_use_map_path
            self.output_path = output_path
            self.date = pd.to_datetime(simulation_date)
            self.transmissivity = transmissivity
            self.start_time = start_time
            self.end_time = end_time
            self.utc = UTC
            self.time = None
            self.shade_data_array = None
            self.land_use_data_array = None
            self.srs = get_srs_from_wkt(get_wkt(self.land_use_map_path))
            print(self.srs)
            self.extents = None
            self.bbox = None
            self.cell_size = 2
            self.no_data_value = -9999

    def _create_aligned_map(self, map_path):
        aligned_map_path = map_path + '_aligned'
        warp(dst=aligned_map_path, src=map_path, dst_srs=self.srs, outputBounds=self.extents,
                        targetAlignedPixels=True, xRes=self.cell_size, yRes=self.cell_size,
                        format='GTiff', dstNodata=self.no_data_value)
        return aligned_map_path

    def _align_maps(self):
        extents = [
            get_extents(self.dsm_map_path),
            get_extents(self.dtm_map_path), 
            get_extents(self.land_use_map_path),
        ]
        x_min = max([map_extents[0] for map_extents in extents])
        y_max = min([map_extents[1] for map_extents in extents])
        x_max = min([map_extents[2] for map_extents in extents])
        y_min = max([map_extents[3] for map_extents in extents])
        self.extents = [x_min, y_min, x_max, y_max]
        self.land_use_map_path = self._create_aligned_map(self.land_use_map_path)
        self.dsm_map_path = self._create_aligned_map(self.dsm_map_path)
        self.dtm_map_path = self._create_aligned_map(self.dtm_map_path)
        land_use_extents = get_extents(self.land_use_map_path)
        self.extents = [land_use_extents[0], land_use_extents[3], land_use_extents[2], land_use_extents[1]]

    def _set_bbox(self):
        xmin, ymin = reproject_points(self.extents[0], self.extents[1],
                                                 src_srs=self.srs, dst_srs='EPSG:4326')
        xmax, ymax = reproject_points(self.extents[2], self.extents[3],
                                                 src_srs=self.srs, dst_srs='EPSG:4326')
        self.bbox = {
            'north': round(ymax + 0.01, 5),
            'south': round(ymin - 0.01, 5),
            'west': round(xmin - 0.01, 5),
            'east': round(xmax + 0.01, 5)
        }

        self.bbox_centroid = {
            "latitude": (self.bbox["north"] + self.bbox["south"]) / 2,
            "longitude": (self.bbox["east"] + self.bbox["west"]) / 2,
            }
        
    def _create_raster_data_array(self, raster_path):
        with rio.open(raster_path) as src:
            raster = src.read(1)
            da = xr.DataArray(raster, dims=('latitude', 'longitude'))
            count_x = len(da.longitude)
            count_y = len(da.latitude)
            x_coord = np.linspace(self.extents[0], self.extents[2], count_x)
            y_coord = np.linspace(self.extents[3], self.extents[1], count_y)
            da = da.assign_coords(latitude=y_coord, longitude=x_coord)
            da = da.where(da != src.nodata, np.nan)
            return da
    

    def _calculate_shade(self):
        # Reproject data
        dsm = self.dsm_data_array.rio.write_crs(self.srs).rio.reproject("EPSG:4326")
        dtm = self.dtm_data_array.rio.write_crs(self.srs).rio.reproject("EPSG:4326")
        land_use = self.land_use_data_array.rio.write_crs(self.srs).rio.reproject("EPSG:4326")

        dsm_building = dsm.where(land_use != 5, dtm)  # Trees replaced by ground
        dsm_tree = dsm.where(land_use == 5, dtm)      # Only trees keep elevation

        dsm_building_array = dsm_building.values
        dsm_tree_array = dsm_tree.values
        dtm_array = dtm.values
        latitudes = dsm.y.values
        longitudes = dsm.x.values

        center_lat = float(latitudes.mean())
        center_lon = float(longitudes.mean())
        dt_utc = pd.Timestamp(self.time).to_pydatetime().replace(tzinfo=timezone(timedelta(hours=self.utc)))
        azimuth = get_azimuth(center_lat, center_lon, dt_utc)
        altitude = get_altitude(center_lat, center_lon, dt_utc)
        print(f"{dt_utc}: Azimuth {azimuth:.2f}°, Altitude {altitude:.2f}°")

        @njit(parallel=True)
        def _compute_shade(dsm_array, azimuth_deg, altitude_deg, cell_size, shadow_value=1):
            rows, cols = dsm_array.shape
            shadow_mask = np.zeros((rows, cols), dtype=np.float32)

            if altitude_deg < 0:
                return np.ones((rows, cols), dtype=np.float32)

            azimuth_rad = radians(azimuth_deg)
            altitude_rad = radians(altitude_deg)
            dx = sin(azimuth_rad)
            dy = -cos(azimuth_rad)
            tan_alt = tan(altitude_rad)
            max_distance = min(int(500 / cell_size), int(50 / tan_alt))

            for r in prange(rows):
                for c in range(cols):
                    z0 = dsm_array[r, c]
                    if isnan(z0) or z0 <= 0:
                        continue

                    for d in range(1, max_distance):
                        x = c + int(d * dx)
                        y = r + int(d * dy)
                        if 0 <= x < cols and 0 <= y < rows:
                            z_line = z0 + d * cell_size * tan_alt
                            z_test = dsm_array[y, x]
                            if not isnan(z_test) and z_test > z_line:
                                shadow_mask[r, c] = shadow_value
                                break
                        else:
                            break
            return shadow_mask

        # Compute building shadows (value = 1.0)
        building_shade = _compute_shade(dsm_building_array, azimuth, altitude, self.cell_size, shadow_value=1.0)
        tree_shade = _compute_shade(dsm_tree_array, azimuth, altitude, self.cell_size, shadow_value=self.transmissivity) 
        combined_shade = np.maximum(building_shade, tree_shade)

        self.shade_data_array = xr.DataArray(
            combined_shade, coords=dsm.coords, dims=dsm.dims, name="shade"
        )
        
        return self.shade_data_array


    def _export_results(self, da, i):
        da = da.fillna(self.no_data_value)
        da = da.rio.write_crs('EPSG:4326').rio.reproject(self.srs)
        da.rio.to_raster(self.output_path +f"shade_at_{i}.00h.tif", driver='GTiff', dtype=np.float32)


    def run(self):
        self._align_maps()
        self._set_bbox()
        self.land_use_data_array = self._create_raster_data_array(self.land_use_map_path)
        self.dsm_data_array = self._create_raster_data_array(self.dsm_map_path)
        self.dtm_data_array = self._create_raster_data_array(self.dtm_map_path)
        for i in range(self.start_time,self.end_time):
            self.time = pd.Timestamp(
                year=self.date.year,
                month=self.date.month,
                day=self.date.day,
                hour=i)
            self.shade_data_array = self._calculate_shade()
            self._export_results(self.shade_data_array, self.srs, i)
