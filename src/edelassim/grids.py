from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS


class GrandesRoussesGrid20m(GSGrid):
    """This grid bound correspond to a bounding box including all mountaineous areas over metropolitan France in WGS84 geographic coordinates."""

    def __init__(self):
        super().__init__(
            x0=736914.0445886956,
            y0=5015138.075400733,
            resolution=20,
            width=1788,
            height=1263,
            crs=CRS.from_epsg(32631),
            name="Sentinel2_GrandesRousses",
        )


class EdelweissGrandesRoussesGrid(GSGrid):
    """This grid bound correspond to a bounding box including all mountaineous areas over metropolitan France in WGS84 geographic coordinates."""

    def __init__(self):
        super().__init__(
            x0=937750,
            y0=6.46425e06,
            resolution=250,
            width=143,
            height=101,
            crs=CRS.from_epsg(2154),
            name="Edelweiss_GrandesRousses",
        )
