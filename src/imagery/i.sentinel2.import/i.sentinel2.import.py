#!/usr/bin/env python3
"""MODULE:      i.sentinel2.import
AUTHOR(S):      Stefan Blumentrath
PURPOSE:        Imports Sentinel-2 satellite data downloaded from e.g. the
                Copernicus Data Space Ecosystem
COPYRIGHT:      (C) 2018-2025 by Stefan Blumentrath
                and the GRASS development team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

# %Module
# % description: Imports Sentinel-2 satellite data downloaded from e.g. the Copernicus Data Space Ecosystem
# % keyword: imagery
# % keyword: satellite
# % keyword: Sentinel
# % keyword: import
# %end

# %option G_OPT_M_DIR
# % key: input
# % description: Name of input directory with downloaded Sentinel data
# % required: yes
# %end

# %option
# % key: product
# % description: ID of the product type to import (default is S2_MSI_L2A)
# % options: S2_MSI_L2A
# % answer: S2_MSI_L2A
# % multiple: no
# % required: yes
# %end

# %option G_OPT_M_DIR
# % key: unzip_dir
# % description: Name of directory into which Sentinel zip-files are extracted (default=input)
# % required: no
# %end

# # %option
# # % key: output_range
# # % description: Range of possible output values for spectral bands (classified and auxilary bands are imported as is)
# # % answer: 0-10000
# # % type: string
# # % guisection: Filter
# # %end

# %option
# % key: bands
# % description: Comma separated list of bands to import (default is all bands)
# % type: string
# % guisection: Filter
# %end

# %option
# % key: file_pattern
# % description: File name pattern to import
# % type: string
# % guisection: Filter
# %end

# %option
# % key: extent
# % type: string
# % required: no
# % multiple: no
# % options: input,region,intersection
# % answer: input
# % description: Output raster map extent
# % descriptions: region;extent of current region;input;extent of input map;intersection;extent of intersection between current region and input map
# % guisection: Filter
# %end

# %option
# % key: memory
# % type: integer
# % required: no
# % multiple: no
# % label: Maximum memory to be used (in MB)
# % description: Cache size for raster rows
# % answer: 300
# %end

# %option G_OPT_F_OUTPUT
# % key: register_output
# % description: Name for output file to use with t.register
# % required: no
# %end

# %option G_OPT_M_DIR
# % key: metadata
# % description: Name of directory into which Sentinel metadata JSON files are saved. Default is PROJECT/MAPSET/cell_misc/MAP_NAME/description.json
# % required: no
# %end

# %option G_OPT_M_COLR
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: i
# % description: Create also an imagery group with the imported raster maps (group names correspond to scene/tile names, e.g. S2A_MSIL2A_20250101T000000_N0511_R008_T32TQM_20250101T000000)
# % guisection: Settings
# %end

# %flag
# % key: l
# % description: Link the raster files using r.external
# % guisection: Settings
# %end

# %flag
# % key: f
# % description: Link the raster files in a fast way, without reading metadata using r.external
# % guisection: Settings
# %end

# %flag
# % key: n
# % description: Force unzipping of archive files
# % guisection: Settings
# %end

# %flag
# % key: o
# % label: Override projection check (use current project's projection)
# % description: Assume that the dataset has same projection as the current project
# % guisection: Settings
# %end

# %flag
# % key: p
# % description: Print raster data to be imported and exit
# % guisection: Print
# %end

# %flag
# % key: r
# % description: Limit import to the computational region
# % guisection: Settings
# %end

# %rules
# % exclusive: -l,-f,-p
# %end

import json
import re
import sys

# import defusedxml.sax.handler as saxhandler
import xml.sax.handler as saxhandler
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import chain
from math import ceil, floor, inf
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE
from typing import TYPE_CHECKING
from xml import sax  # NOQA: S406
from zipfile import ZipFile

import grass.script as gs
import numpy as np
from grass.pygrass.gis.region import Region
from grass.pygrass.modules import Module, MultiModule

if TYPE_CHECKING:
    from osgeo import gdal, osr

BAND_SUFFIX = ".jp2"
RESAMPLE_DICT = {
    "nearest": "near",
    "bilinear": "bilinear",
    "bicubic": "cubic",
    "cubicspline": "cubicspline",
    "lanczos": "lanczos",
    "average": "average",
    "mode": "mode",
    "max": "max",
    "min": "min",
    "med": "med",
    "Q1": "Q1",
    "Q3": "Q3",
}
ALIGN_REGION = None
BANDS = None
PRODUCT_DEFAULTS = {
    "S2_MSI_L2A": {
        "filter": ".*S2.*MSIL2A.*",
        "bands": {
            "MSK_CLDPRB_20m",
            "MSK_SNWPRB_20m",
            "SCL_20m",
            "AOT",
            "WVP",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        },
    },
}


def get_band_info() -> dict:
    """Populate the global BANDS dictionary with band information.

    Needs to be a function because of lazy import of `osgeo.gdal`.

    Returns
    -------
    bands: dict
        Dictionary containing band information.

    """
    return {
        # AUX and QI bands
        "MSK_CLDPRB_20m": {
            "file_path": f"MSK_CLDPRB_20m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_cloud_probability",
        },
        "MSK_SNWPRB_20m": {
            "file_path": f"MSK_CLDPRB_20m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_snow_probability",
        },
        "MSK_CLDPRB_60m": {
            "file_path": f"MSK_CLDPRB_60m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_cloud_probability_60m",
        },
        "MSK_SNWPRB_60m": {
            "file_path": f"MSK_SNWPRB_60m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_snow_probability_60m",
        },
        "AOT": {
            "file_path": f"AOT_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_AOT",
        },
        "WVP": {
            "file_path": f"WVP_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_WVP",
        },
        "AOT_20m": {
            "file_path": f"AOT_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_AOT_20m",
        },
        "WVP_20m": {
            "file_path": f"WVP_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_WVP_20m",
        },
        "SCL_20m": {
            "file_path": f"SCL_20m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_SCL",
        },
        "AOT_60m": {
            "file_path": f"AOT_60m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_AOT_60m",
        },
        "WVP_60m": {
            "file_path": f"WVP_60m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_WVP_60m",
        },
        "SCL_60m": {
            "file_path": f"SCL_60m{BAND_SUFFIX}",
            "resample": "nearest",
            "data_type": gdal.GDT_Byte,
            "id": None,
            "semantic_label": "S2_SCL_60m",
        },
        # Spectral bands
        "B1": {
            "file_path": f"B01_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "0",
            "semantic_label": "S2_1",
        },
        "B2": {
            "file_path": f"B02_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "1",
            "semantic_label": "S2_2",
        },
        "B3": {
            "file_path": f"B03_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "2",
            "semantic_label": "S2_3",
        },
        "B4": {
            "file_path": f"B04_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "3",
            "semantic_label": "S2_4",
        },
        "B5": {
            "file_path": f"B05_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "4",
            "semantic_label": "S2_5",
        },
        "B6": {
            "file_path": f"B06_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "5",
            "semantic_label": "S2_6",
        },
        "B7": {
            "file_path": f"B07_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "6",
            "semantic_label": "S2_7",
        },
        "B8": {
            "file_path": f"B08_10m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "7",
            "semantic_label": "S2_8",
        },
        "B8A": {
            "file_path": f"B8A_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "8",
            "semantic_label": "S2_8A",
        },
        "B9": {
            "file_path": f"B09_60m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "9",
            "semantic_label": "S2_9",
        },
        "B10": {
            "file_path": f"B10_60m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "10",
            "semantic_label": "S2_10",
        },
        "B11": {
            "file_path": f"B11_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "11",
            "semantic_label": "S2_11",
        },
        "B12": {
            "file_path": f"B12_20m{BAND_SUFFIX}",
            "resample": "bilinear",
            "data_type": gdal.GDT_UInt16,
            "id": "12",
            "semantic_label": "S2_12",
        },
    }


class TreeBuilder(saxhandler.ContentHandler):
    """Build a tree from the XML file."""

    def __init__(
        self,
        requested_keys: tuple = (
            # from MTD_MSIL2A
            "Product_Info",
            # "n1:General_Info",
            "Scene_Classification_List",
            "Image_Content_QI",
            "Quality_Inspections",
            "BOA_ADD_OFFSET_VALUES_LIST",
            "QUANTIFICATION_VALUES_LIST",
            "Special_Values",
            "Granule_List",
            # from MTD_TL
            "Tile_Geocoding",
            "Mean_Viewing_Incidence_Angle_List",
            "Sun_Angles_Grid",  # Might be needed for L1C data
            "Mean_Sun_Angle",
        ),
    ) -> None:
        """Initialize the tree builder.

        Parameters
        ----------
        requested_keys: tuple
            The keys to be extracted from the XML file

        """
        self.name = None
        self.attrs = None
        self.keys = []
        self.requested_keys = requested_keys
        self.elements = []
        self._dict = {k: [] for k in requested_keys}

    def startElement(self, name: str, attrs: list | str) -> None:
        """Identify the start of an element.

        Parameters
        ----------
        name: str
            The name of the element
        attrs: list | str
            The attributes of the element

        """
        self.name = name
        self.attrs = attrs
        self.keys.append(name)

    def endElement(self, name: str) -> None:
        """Identify the end of an element.

        Parameters
        ----------
        name: str
            The name of the element

        """
        self.keys.remove(name)

    def characters(self, content: str) -> list:
        """Get the content of the element.

        Parameters
        ----------
        content: str
            The content of the element

        Returns
        -------
        list
            A list with the requested key-value pairs

        """
        content = content.strip()
        attrs = "" if not self.attrs.items() else self.attrs.items()
        intersect = list(set(self.keys).intersection(self.requested_keys))
        if content and intersect:
            self.elements.append(
                [intersect[0], {self.name: (attrs, content) if attrs else content}],
            )
            self._dict[intersect[0]].append(
                {self.name: (attrs, content) if attrs else content},
            )
        return [self.keys[0:-1], {self.name: (attrs, content) if attrs else content}]


def get_key_value_pairs(builder: TreeBuilder, parent_key: str) -> dict:
    """Get key-value pairs from the XML tree.

    Parameters
    ----------
    builder: TreeBuilder
        The tree builder object
    parent_key: str
        The parent key to get the values from

    Returns
    -------
    dict
        The key-value pairs

    """
    values_list = [list(p.values())[0] for p in builder._dict.get(parent_key)]
    values_dict = {}
    while values_list:
        key = values_list.pop(0)
        val = values_list.pop(0)
        values_dict[key] = val
    return values_dict


def get_scene_metadata(scene_path: Path, product_type: str = "S2_MSI_L2A") -> dict:
    """Get scene metadata from XML file(s).

    Parameters
    ----------
    scene_path: Path
        Path to the scene directory
    product_type: str
        The product type of the scene

    Returns
    -------
    scene_metadata: dict
        A dict with the scene metadata (JSON)

    """
    builder = TreeBuilder()
    for mtd_file in scene_path.glob("**/MTD_*.xml"):
        sax.parseString(mtd_file.read_text(), builder)  # NOQA: S317

    scene_metadata = {}
    if product_type == "S2_MSI_L2A":
        # Get the product info
        scene_metadata["product_metadata"] = {
            k: v for d in builder._dict.get("Product_Info") for k, v in d.items()
        }
        # get the image files
        scene_metadata["granule_list"] = [
            v["IMAGE_FILE"] for v in builder._dict["Granule_List"]
        ]
        # Get the special values (NODATA, SATURATED, etc.)
        scene_metadata["special_values"] = get_key_value_pairs(
            builder,
            "Special_Values",
        )
        # Get the quantification values
        scene_metadata["quantification_values"] = {
            k: float(v[1])
            for d in builder._dict.get("QUANTIFICATION_VALUES_LIST")
            for k, v in d.items()
        }
        # Get the add offset values
        scene_metadata["boa_offset_values"] = {
            v[0][0][1]: float(v[1])
            for d in builder._dict.get("BOA_ADD_OFFSET_VALUES_LIST")
            for k, v in d.items()
        }
        # Get the scene classification list
        scene_metadata["scene_classification"] = get_key_value_pairs(
            builder,
            "Scene_Classification_List",
        )
        # Get the CRS
        scene_metadata["CRS_EPSG"] = int(
            [
                d.get("HORIZONTAL_CS_CODE").split(":")[1]
                for d in builder._dict.get("Tile_Geocoding")
                if d.get("HORIZONTAL_CS_CODE")
            ][0],
        )
        # Get the image content quality inspection
        scene_metadata["product_metadata"].update(
            {k: v for d in builder._dict.get("Image_Content_QI") for k, v in d.items()},
        )
        # Get the mean sun angles
        scene_metadata["product_metadata"].update(
            {
                f"MEAN_SUN_{k}": float(v[1])
                for d in builder._dict.get("Mean_Sun_Angle")
                for k, v in d.items()
            },
        )
        # Get quality inspections
        scene_metadata["product_metadata"].update(
            {
                v[0][0][1]: v[1]
                for d in builder._dict.get("Quality_Inspections")
                for k, v in d.items()
            },
        )
    return scene_metadata


def transform_bounding_box(
    bbox: tuple[float, float, float, float],
    transform,
    edge_densification: int = 15,
) -> tuple:
    """Transform the datasets bounding box into the projection of the project.

    Edges are densified. bbox is a tuple of (xmin, ymin, xmax, ymax)
    Adapted from:
    https://gis.stackexchange.com/questions/165020/how-to-calculate-the-bounding-box-in-projected-coordinates

    Parameters
    ----------
    bbox: tuple
        The bounding box to transform
    transform: osr.CoordinateTransformation
        The transformation object to be used for the coordinate transformation
    edge_densification: int
        The number of points to densify the edges with (Default value = 15)

    Returns
    -------
    bbox: tuple
        The transformed bounding box

    """
    u_l = np.array((bbox[0], bbox[3]))
    l_l = np.array((bbox[0], bbox[1]))
    l_r = np.array((bbox[2], bbox[1]))
    u_r = np.array((bbox[2], bbox[3]))

    def _transform_vertex(vertex: tuple[float, float]) -> tuple[float, float]:
        """Transform the coordinates of a vertex to the new coordinate system.

        Parameters
        ----------
        vertex: tuple
            Coordinates of the vertex to transform

        Returns
        -------
        transformed_vertex: tuple
            Coordinates of the transformed vertex

        """
        try:
            x_transformed, y_transformed, _ = transform.TransformPoint(*vertex)
        except Exception:
            x_transformed, y_transformed = inf, inf
        return (x_transformed, y_transformed)

    # This list comprehension iterates over each edge of the bounding box,
    # divides it into `edge_densification` number of points, then reduces
    # that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    return [
        bounding_fn(
            [
                _transform_vertex(p_a * v + p_b * (1 - v))
                for v in np.linspace(0, 1, edge_densification)
            ],
        )
        for p_a, p_b, bounding_fn in [
            (u_l, l_l, lambda point_list: min(p[0] for p in point_list)),
            (l_l, l_r, lambda point_list: min(p[1] for p in point_list)),
            (l_r, u_r, lambda point_list: max(p[0] for p in point_list)),
            (u_r, u_l, lambda point_list: max(p[1] for p in point_list)),
        ]
    ]


def check_projection_match(reference_crs: str, s2_tile_epsg: int) -> bool:
    """Check if project projections matches projection of S2 tile EPSG code.

    Using gdal/osr

    Parameters
    ----------
    reference_crs: str
        WKT string of the reference CRS
    s2_tile_epsg: int
        EPSG code of the S2 tile

    Returns
    -------
    bool

    """
    tile_crs = osr.SpatialReference()
    tile_crs.ImportFromEPSG(s2_tile_epsg)
    project_crs = osr.SpatialReference()
    project_crs.ImportFromWkt(reference_crs)
    return tile_crs.IsSame(project_crs)


def align_windows(window: dict, region: Region | None = None) -> dict:
    """Align two regions.

    Python version of:
    https://github.com/OSGeo/grass/blob/main/lib/raster/align_window.c

    Modifies the input ``window`` to align to ``region``. The
    resolutions in ``window`` are set to match those in ``region``
    and the ``window`` edges (north, south, east, west) are modified
    to align with the grid of the ``region``.

    The ``window`` may be enlarged if necessary to achieve the
    alignment. The north is rounded northward, the south southward,
    the east eastward and the west westward. Lon-lon constraints are
    taken into consideration to make sure that the north doesn't go
    above 90 degrees (for lat/lon) or that the east does "wrap" past
    the west, etc.

    Parameters
    ----------
    window: dict
        A dict with the window to align, with keys north, south, east,
        west, nsres, ewres, is_latlong
    region: Region
        A GRASS GIS Region object to align to, with keys north, south,
        east, west, nsres, ewres, is_latlong

    Returns
    -------
    aligned_window: dict
        A modified version of ``window`` that is aligend to ``reegion``

    """
    aligned_window = {
        "nsres": region.nsres,
        "ewres": region.ewres,
        "is_latlong": region.proj == "ll",
        "north": (
            region.north
            if window[3] == inf
            else (
                region.north
                - floor((region.north - window[3]) / region.nsres) * region.nsres
            )
        ),
        "south": (
            region.south
            if window[1] == inf
            else (
                region.south
                - ceil((region.south - window[1]) / region.nsres) * region.nsres
            )
        ),
        "west": (
            region.west
            if window[0] == inf
            else (
                region.west
                + floor((window[0] - region.west) / region.ewres) * region.ewres
            )
        ),
        "east": (
            region.east
            if window[2] == inf
            else (
                region.east
                + ceil((window[2] - region.east) / region.ewres) * region.ewres
            )
        ),
    }
    if aligned_window["is_latlong"]:
        while aligned_window["north"] > 90.0 + aligned_window["nsres"] / 2.0:
            aligned_window["north"] -= aligned_window["nsres"]
        while aligned_window["south"] < -90.0 - aligned_window["nsres"] / 2.0:
            aligned_window["south"] += aligned_window["nsres"]
    return aligned_window


def legalize_name_string(string: str) -> str:
    """Replace conflicting characters with _.

    Parameters
    ----------
    string : str
        String to be transformed to a legal map name

    Returns
    -------
    legal_map_name: str
        Legal map name

    """
    return re.sub(r"[^\w\d-]+|[^\x00-\x7F]+|[ -/\\]+", "_", string)


class Sentinel2Importer:
    """Class to import Sentinel-2 data into GRASS GIS."""

    def __init__(
        self,
        input_dir: Path,
        unzip_dir: Path,
        *,
        selected_bands: list | None = None,
        projection_wkt: str | None = None,
        band_filter: str | None = None,
        print_only: bool = False,
        reproject: bool = False,
        link: bool = False,
        group: bool = False,
        override: bool = False,
        nprocs: int = 1,
    ) -> None:
        """Initialize the Sentinel2Importer class."""
        # list of directories & maps to cleanup
        self.input_dir = None
        self.metadata_dicts = {}
        self.register_strings = []
        self.module_list = []
        self.print_info = []
        self.zip_archives = set()
        self.safe_files = set()
        self.selected_bands = {b: BANDS.get(b) for b in selected_bands}
        self.reference_crs = projection_wkt
        self.band_filter = band_filter
        self.reproject = reproject
        self.group = group
        self.link = link
        self.override = override
        self.nprocs = nprocs
        self.print_only = print_only
        self.mapset = gs.gisenv()["MAPSET"]

        # Check selected bands
        for band in list(self.selected_bands):
            if self.selected_bands.get(band) is None:
                gs.warning(
                    _("Band <{band}> not supported. Use one of <{bands}>.").format(
                        band=band,
                        bands=", ".join(BANDS.keys()),
                    ),
                )
                self.selected_bands.pop(band)

        # check if input dir exists
        if input_dir:
            if input_dir.exists():
                self.input_dir = input_dir
            else:
                gs.fatal(
                    _("Input directory <{}> does not exist").format(str(input_dir)),
                )

        # check if unzip dir exists
        if not unzip_dir:
            unzip_dir = input_dir

        self.unzip_dir = unzip_dir
        try:
            unzip_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            gs.fatal(_("Directory <{}> not accessible").format(unzip_dir))

        # Setup import module objects
        import_flags = "oa" if flags["o"] else "a"
        kwargs = {"quiet": True, "run_": False, "finish_": False}
        import_kwargs = kwargs.copy()
        import_module_name = "r.external"
        if flags["f"]:
            import_flags += "r"
        elif flags["l"]:
            import_flags += "m"
        else:
            if flags["r"]:
                import_flags += "r"
            import_module_name = "r.in.gdal"
            import_kwargs["memory"] = options["memory"]

        import_module = Module(
            import_module_name,
            overwrite=gs.overwrite(),
            flags=import_flags,
            **import_kwargs,
        )
        self.import_modules = {
            "import": import_module,
            "timestamp": Module("r.timestamp", **kwargs),
            "colors": (
                Module(
                    "r.colors",
                    color=options["color"],
                    **kwargs,
                )
                if options["color"]
                else None
            ),
            "support": Module("r.support", **kwargs),
            "categories": Module("r.category", separator=":", rules="-", **kwargs),
        }

    def __del__(self) -> None:
        """Cleanup temporary files."""
        """For map in self._map_list:
            if gs.find_file(map, element="cell", mapset=".")["file"]:
                gs.run_command(
                    "g.remove", flags="fb", type="raster", name=map, quiet=True
                )
            if gs.find_file(map, element="vector", mapset=".")["file"]:
                gs.run_command(
                    "g.remove", flags="f", type="vector", name=map, quiet=True
                )

        if flags["l"]:
            # unzipped files are required when linking
            return

        # otherwise unzipped directory can be removed (?)
        for dirname in self._dir_list:
            dirpath = self.unzip_dir / dirname
            gs.debug("Removing <{}>".format(str(dirpath)))
            try:
                shutil.rmtree(dirpath)
            except OSError:
                pass
        """

    def create_vrt(
        self,
        product_path: Path,
        product_name: str,
        gisenv: dict,
        *,
        resample: str = "nearest",
        nodata: list | tuple | None = (0, 65355),
        rescale: bool = False,
        scale: float = 1.0,
        offset: float = 0.0,
        data_type: int | None = None,
        equal_proj: bool = True,
        transform: bool = True,
        region_cropping: bool = False,
        recreate: bool = False,
    ) -> str:
        """Create a GDAL VRT for import.

        kwargs:
        - resample: str
        - nodata: int
        - rescale: bool
        - scale: float
        - offset: float
        - equal_proj: bool
        - transform: bool
        - region_cropping: bool
        - recreate: bool
        Offset needs to be applied separate (two steps) or as rescaled values
        band.SetScale(1.0 / 10000.0)
        band.SetOffset(-1000.0 / 10000.0)
        to = gdal.TranslateOptions(
            noData=65535,
            format="VRT",
            unscale=True,
            outputType=gdal.GDT_Float32,
            )
        dsv = gdal.Translate("/tmp/test_b.vrt", ds, options=to)
        dsv = None
        !gdalinfo "/tmp/test_b.vrt" -stats
        Scale should not be applied at all (keep GDT_Int16)
        We need two VRTs, one for the offset and one for warping
        (if needed)

        Returns:
        vrt_path: str
            The path to the created VRT file.

        """
        # Apply Offset (and Scale if needed)
        kwargs = {
            "format": "VRT",
            "unscale": True,
            "resampleAlg": resample,
        }
        if data_type is None:
            data_type = gdal.GDT_Int16

        with gdal.Open(str(product_path)) as ds:
            band = ds.GetRasterBand(1)
            if nodata:
                band.SetNoDataValue(nodata)
                kwargs["noData"] = nodata
            if rescale and scale and offset:
                band.SetScale(1.0 / scale)
                band.SetOffset(offset / scale)
                data_type = gdal.GDT_Float32
            elif offset:
                band.SetOffset(offset)
            kwargs["outputType"] = data_type

            vrt_dir = Path(gisenv["GISDBASE"]).joinpath(
                gisenv["LOCATION_NAME"],
                gisenv["MAPSET"],
                "gdal",
            )
            vrt_offset = vrt_dir / f"{legalize_name_string(product_name)}_offset.vrt"
            vrt_offset_name = str(vrt_offset)
            if vrt_offset.exists() and not recreate:
                return vrt_offset_name

            # if region_cropping:
            #     aligned_bbox = ALIGN_REGION(transformed_bbox)

            vrt_offset = gdal.Translate(
                vrt_offset_name,
                ds,  # Use already opened dataset here
                options=gdal.TranslateOptions(
                    **kwargs,
                    # stats=True,
                    # outputBounds=
                ),
            )
            vrt_offset = None
            if equal_proj:
                return vrt_offset_name

            project_crs = osr.SpatialReference()
            project_crs.ImportFromWkt(self.reference_crs)
            dataset_crs = ds.GetSpatialRef()
            transform = osr.CoordinateTransformation(dataset_crs, project_crs)

            vrt = vrt_dir / f"{legalize_name_string(product_name)}.vrt"

            vrt_name = str(vrt)

            gt = ds.GetGeoTransform()
            transformed_bbox = transform_bounding_box(
                (
                    gt[0],
                    gt[3] + gt[5] * ds.RasterYSize,
                    gt[0] + gt[1] * ds.RasterXSize,
                    gt[3],
                ),
                transform,
                edge_densification=15,
            )
            kwargs = {
                "dstSRS": self.reference_crs,
                "format": "VRT",
                "resampleAlg": resample,
                "outputType": data_type,
            }
            if nodata is not None:
                kwargs["srcNodata"] = nodata
            # Resolution should be probably taken from region rather than from source dataset
            # Cropping to computational region should only be done with r-flag
            if region_cropping:
                aligned_bbox = ALIGN_REGION(transformed_bbox)
                kwargs["xRes"] = aligned_bbox["ewres"]  # gt[1]
                kwargs["yRes"] = aligned_bbox["nsres"]  # -gt[5]
                kwargs["outputBounds"] = (
                    aligned_bbox["west"],
                    aligned_bbox["south"],
                    aligned_bbox["east"],
                    aligned_bbox["north"],
                )

            vrt = gdal.Warp(
                vrt_name,
                vrt_offset_name,
                options=gdal.WarpOptions(
                    **kwargs,
                    # outputType=gdal.GDT_Int16,
                ),
            )
            vrt = None
        return vrt_name

    def _unzip(self, file_path: str) -> None:
        """Unzip a single zip file.

        Parameters
        ----------
        file_path : str
            The path to the file to unzip.

        """
        # extract all zip files from input directory
        gs.verbose(_("Unziping <{}>...").format(file_path))
        with ZipFile(file_path) as fd:
            fd.extractall(path=self.unzip_dir)

    def unzip(
        self,
        *,
        file_pattern: str | None = None,
        force: bool | None = False,
    ) -> None:
        """Unzip zip files in input directory with pattern matching.

        Parameters
        ----------
        file_pattern : str, optional
            The pattern to match the zip files.
        force : bool, optional
            If True, force the extraction of all zip files.

        """
        # Filter zip files from input directory
        input_files = self.input_dir.glob("S2*.zip")
        pattern = None
        if file_pattern:
            pattern = re.compile(rf".*{file_pattern}.*.zip")

        for file_path in input_files:
            if pattern and not pattern.match(str(file_path)):
                continue
            safe = self.unzip_dir / file_path.stem
            if force or not (safe.exists() or safe.with_suffix(".SAFE").exists()):
                self.zip_archives.add(file_path)
        # Unzip archives in parallel
        nprocs = min(len(self.zip_archives), self.nprocs)
        if nprocs > 1:
            with Pool(self.nprocs) as pool:
                pool.map(self._unzip, self.zip_archives)
        else:
            for archive in self.zip_archives:
                self._unzip(archive)

    def filter_safe_files(self, *, file_pattern: str | None = None) -> None:
        """Filter SAFE files from unzipped directory with pattern matching.

        Parameters
        ----------
        file_pattern : str, optional
            The pattern to match the SAFE files.

        """
        pattern = None
        if file_pattern:
            pattern = re.compile(f".*{file_pattern}.*.SAFE")

        for safe in self.unzip_dir.glob("S2*.SAFE"):
            if pattern and not pattern.match(str(safe)):
                continue
            self.safe_files.add(safe)

        if len(self.safe_files) < 1:
            gs.fatal(
                _(
                    "No Sentinel files found to import in directory <{}>. Please check input and pattern_file options.",
                ).format(str(self.unzip_dir)),
            )

    @staticmethod
    def _check_project_projection_meters() -> bool:
        """Check if project projection uses meters.

        Returns
        -------
        bool

        """
        units = gs.parse_command("g.proj", flags="g")["units"]
        return units.lower() == "meters"

    def filter_bands(self, pattern: str | None = None) -> None:
        """Filter bands from SAFE files.

        Parameters
        ----------
        pattern : str, optional
            Pattern to filter bands.

        """
        # Need to investigate if product level dependent
        # filter_p = r".*{}.*.jp2".format(pattern) if pattern else r".*_B.*.jp2$|.*_SCL*.jp2$"
        filter_p = (
            rf".*{pattern}.*.jp2"
            if pattern
            else r".*(MSK_|_B[0-9]|_WVP|_AOT|_SCL).*0m.jp2$"
        )

        gs.debug(_("Filter: {}").format(filter_p), 1)
        self.files = self._filter(filter_p, force_unzip=not flags["n"])

    def _prepare_product_import(self, safe: Path) -> tuple:
        """Prepare import of Sentinel-2 products.

        Parameters
        ----------
        safe : Path
            Path to SAFE directory to prepare to import.


        """
        scene = safe.stem
        module_list = []
        register_strings = []
        gs.verbose(_("Preparing import of scene <{}>...").format(scene))

        # Get Metadata
        scene_metadata = get_scene_metadata(safe, product_type="S2_MSI_L2A")
        start_time = scene_metadata["product_metadata"][
            "DATATAKE_SENSING_START"
        ].rstrip("Z")

        projection_matches = bool(
            check_projection_match(self.reference_crs, scene_metadata["CRS_EPSG"]),
        )
        import_dict = {}
        print_output = []
        result_maps = []

        # Filter bands
        bands = list(safe.glob(f"**/*{BAND_SUFFIX}"))
        for band, band_config in self.selected_bands.items():
            matched_bands = [
                b for b in bands if str(b).endswith(band_config["file_path"])
            ]
            if not matched_bands:
                gs.warning(_("Band <{}> not found in scene <{}>.").format(band, scene))
                continue
            jp2 = matched_bands[0]
            semantic_label = BANDS[band]["semantic_label"]
            product_name = f"{scene}.{semantic_label}"
            if (
                gs.find_file(product_name, mapset=self.mapset)["fullname"]
                and not gs.overwrite()
            ):
                gs.warning(
                    _(
                        "Product <{}> already exists in mapset <{}>. Skipping import.",
                    ).format(product_name, self.mapset),
                )
                continue
            if self.print_only:
                print_output.append(
                    "|".join(
                        [
                            product_name,
                            str(jp2),
                            semantic_label or "",
                            start_time,
                            str(scene_metadata["CRS_EPSG"]),
                        ],
                    ),
                )
                continue
            # Implicitly handling processing baseline
            # Baseline < 05.00 do not have boa offset values
            # Needs to be checked for L1C products
            # Newer baselines > 05.11 should probably be handled flaged with a warning
            nodata = None
            offset = None
            scale = None
            boa_offset_values = scene_metadata.get("boa_offset_values", {})
            quantification_values = scene_metadata.get("quantification_values", {})
            if band.startswith("B"):
                nodata = float(list(scene_metadata["special_values"].values())[-1])
                offset = float(boa_offset_values.get(BANDS[band]["id"], 0.0))
                scale = float(
                    quantification_values.get("BOA_QUANTIFICATION_VALUE", 1.0),
                )
            elif band.startswith("AOT"):
                scale = float(
                    quantification_values.get("AOT_QUANTIFICATION_VALUE", 1.0),
                )
            elif band.startswith("WVP"):
                scale = float(
                    quantification_values.get("WVP_QUANTIFICATION_VALUE", 1.0),
                )
            resampling = BANDS[band]["resample"]
            # Create VRT
            vrt = self.create_vrt(
                jp2,
                product_name,
                gs.gisenv(),
                resample=resampling,
                nodata=nodata,
                rescale=False,
                scale=scale,
                offset=offset,
                data_type=BANDS[band]["data_type"],
                equal_proj=projection_matches,
                region_cropping=True,
                recreate=gs.overwrite(),
            )
            import_modules = deepcopy(self.import_modules)
            import_modules["import"].inputs.input = vrt
            import_modules["import"].outputs.output = product_name
            import_modules["timestamp"].inputs.map = product_name
            import_modules["timestamp"].inputs.date = datetime.fromisoformat(
                start_time,
            ).strftime("%d %b %Y %H:%M:%S.%f")
            if import_modules["colors"]:
                import_modules["colors"].inputs.map = product_name
            else:
                import_modules["colors"] = None
            if import_modules["categories"] and band.startswith(("SCL", "MSK_CLASSI")):
                import_modules["categories"].inputs.map = product_name
                import_modules["categories"].inputs["stdin"].value = "\n".join(
                    [
                        f"{cat}:{label}"
                        for label, cat in scene_metadata["scene_classification"].items()
                    ],
                )
            else:
                import_modules["categories"] = None
            import_modules["support"].inputs.map = product_name
            import_modules["support"].inputs.semantic_label = semantic_label

            module_list.append(
                [
                    import_modules[module]
                    for module in [
                        "import",
                        "timestamp",
                        "colors",
                        "support",
                        "categories",
                    ]
                    if import_modules[module]
                ],
            )
            register_strings.append(
                f"{product_name}@{self.mapset}|{start_time}|{semantic_label}",
            )
            result_maps.append(f"{product_name}@{self.mapset}")
        import_dict[scene] = {
            "metadata": scene_metadata,
            "result_maps": result_maps,
        }

        if self.print_only:
            return print_output

        if not import_dict:
            gs.fatal(
                _(
                    "No bands files found to import in directory <{}>. Please check input and pattern options.",
                ).format(str(self.unzip_dir)),
            )
        return import_dict, module_list, register_strings

    def prepare_product_import(self) -> None:
        """Prepare import of Sentinel-2 products."""
        nprocs = min(len(self.safe_files), self.nprocs)
        if nprocs > 1:
            with Pool(nprocs) as pool:
                product_imports = pool.map(
                    self._prepare_product_import,
                    self.safe_files,
                )
        else:
            product_imports = [
                self._prepare_product_import(sf) for sf in self.safe_files
            ]
        if self.print_only:
            print("\n".join(sorted(chain(*product_imports))))
            sys.exit(0)

        for element in product_imports:
            metadata_dicts, module_list, register_strings = element
            self.register_strings.extend(register_strings)
            self.metadata_dicts.update(metadata_dicts)
            self.module_list.extend(module_list)

    @staticmethod
    def _run_product_import(multi_module: list) -> None:
        """Run import of Sentinel-2 product.

        Parameters
        ----------
        multi_module: list
            List of GRASS GIS Module objects to run in sequence

        """
        multi_module = MultiModule(multi_module)
        multi_module.run()

    def run_product_import(self) -> None:
        """Run import of Sentinel-2 products."""
        nprocs = min(len(self.module_list), self.nprocs)
        if nprocs > 1:
            with Pool(nprocs) as pool:
                pool.map(self._run_product_import, *[self.module_list])
        else:
            for mod in self.module_list:
                gs.debug(mod[-1].get_bash())
                self._run_product_import(mod)

    def print_products(self) -> None:
        """Print list of products to import."""
        for f in self.safe_files:
            print(
                f"{f} {1 if self._check_projection(f) else 0}"
                f" (EPSG: {self._raster_epsg(f)})\n",
            )

    def write_metadata(self) -> None:
        """Write metadata for maps."""
        gs.verbose(_("Writing metadata to maps..."))
        env = gs.gisenv()
        json_folder = (
            Path(env["GISDBASE"]) / env["LOCATION_NAME"] / env["MAPSET"] / "cell_misc"
        )
        if options["metadata"]:
            json_folder = Path(options["metadata"])

        for result_map, meta_dict in self.metadata_dicts.items():
            metadatajson = json_folder / result_map
            metadatajson.mkdir(parents=True, exist_ok=True)
            (metadatajson / "description.json").write_text(json.dumps(meta_dict))

    def create_scene_groups(self, scene_id: str, scene_metadata: list) -> None:
        """Create imagery groups for imported products."""
        # group_maps = sorted(scene_metadata["result_maps"])
        if gs.find_file(name=scene_id, element="group")["file"] and gs.overwrite():
            print("removing ", scene_id)
            Module(
                "g.remove",
                flags="f",
                type="group",
                name=scene_id,
                quiet=True,
                stderr_=PIPE,
            )
        print(scene_metadata["result_maps"])
        print(sorted(scene_metadata["result_maps"]))
        scene_maps = sorted(scene_metadata["result_maps"])
        print(scene_maps)
        Module(
            "i.group",
            group=scene_id,
            input=",".join(scene_maps),
            quiet=True,
            # overwrite=True,
        )

    def create_register_file(self, filename: str, scene_metadata: bool = False) -> None:
        """Create a file for use with t.register.

        Parameters
        ----------
        filename: str
            Path to the file to create
        scene_metadata: bool
            If True, include scene metadata in the register file

        """
        if not scene_metadata:
            gs.verbose(_("Creating register file <{}>...").format(filename))
            Path(filename).write_text(
                "\n".join(self.register_strings) + "\n",
                encoding="utf-8",
            )
            return
        gs.verbose(
            _("Creating register file <{}> including scene metadata...").format(
                filename,
            ),
        )
        Path(filename).write_text(
            json.dumps(
                self.metadata_dicts
                | {"register_strings": "\n".join(self.register_strings) + "\n"},
                indent=2,
            ),
            encoding="utf-8",
        )
        return


def main() -> None:
    """Import using Sentinel2Importer."""
    # Get GRASS GIS environment info
    grass_env = dict(gs.gisenv())

    # Get BANDS info
    global BANDS
    BANDS = get_band_info()

    # initialize file filter pattern
    file_filter_pattern = PRODUCT_DEFAULTS[options["product"]]["filter"]
    if options["file_pattern"]:
        file_filter_pattern = options["file_pattern"]

    # Create directory for vrt files if needed
    if flags["l"] or flags["f"] or flags["r"]:
        vrt_dir = Path(grass_env["GISDBASE"]).joinpath(
            grass_env["LOCATION_NAME"],
            grass_env["MAPSET"],
            "gdal",
        )
    else:
        vrt_dir = Path(gs.tempfile(create=False))
    if not vrt_dir.is_dir():
        vrt_dir.mkdir()

    # Current region
    global ALIGN_REGION
    ALIGN_REGION = partial(align_windows, region=Region())

    importer = Sentinel2Importer(
        Path(options["input"]),
        Path(options["unzip_dir"]),
        projection_wkt=gs.read_command("g.proj", flags="wf").strip(),
        selected_bands=(
            options["bands"].split(",")
            if options["bands"]
            else PRODUCT_DEFAULTS[options["product"]]["bands"]
        ),
        print_only=flags["p"],
        reproject=True,
        link=flags["l"] or flags["f"],
        group=flags["i"],
        override=flags["o"],
        nprocs=int(options["nprocs"]),
    )
    importer.unzip(file_pattern=file_filter_pattern, force=flags["n"])
    importer.filter_safe_files(file_pattern=file_filter_pattern)
    importer.prepare_product_import()
    importer.run_product_import()

    if flags["i"]:
        print("adding group module")
        if importer.nprocs > 1:
            with Pool(importer.nprocs) as pool:
                pool.starmap(
                    importer.create_scene_groups, *[importer.metadata_dicts.items()]
                )
        else:
            for scene_id, scene_dict in importer.metadata_dicts.items():
                importer.create_scene_groups(scene_id, scene_dict)

    importer.write_metadata()

    if options["register_output"]:
        # create t.register file if requested
        importer.create_register_file(options["register_output"], flags["i"])


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    try:
        from osgeo import gdal, ogr, osr

        gdal.UseExceptions()
        ogr.UseExceptions()
    except ImportError as e:
        gs.fatal(_("Unable to load GDAL Python bindings: {}").format(e))

    sys.exit(main())
