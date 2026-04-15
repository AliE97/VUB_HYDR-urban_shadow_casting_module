from osgeo import gdal as osgeo_gdal, ogr, osr
from pyproj import Transformer, CRS


def warp(src, dst, dst_srs=None, **kwargs):
    osgeo_gdal.Warp(destNameOrDestDS=dst, srcDSOrSrcDSTab=src, dstSRS=dst_srs, **kwargs)

def get_extents(map_path):
    ds = osgeo_gdal.Open(map_path)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    lrx = ulx + (ds.RasterXSize * xres)
    lry = uly + (ds.RasterYSize * yres)
    del ds
    return ulx, uly, lrx, lry  # west, north, east, south

def reproject_points(x, y, src_srs, dst_srs):
    transformer = Transformer.from_crs(src_srs, dst_srs, always_xy=True)
    return transformer.transform(x, y)

def get_srs_from_wkt(wkt):
    return CRS(wkt).to_string()

def get_wkt(map_path):
    ds = osgeo_gdal.Open(map_path)
    pr = ds.GetProjection()
    sr = osr.SpatialReference(pr)
    wkt = sr.ExportToWkt()
    del ds
    return wkt