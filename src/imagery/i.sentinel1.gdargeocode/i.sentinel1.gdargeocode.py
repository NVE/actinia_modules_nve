#!/usr/bin/env python3

"""
MODULE:       i.sentinel_1.gdargeocode
AUTHOR(S):    Stefan Blumentrath
PURPOSE:      Searches and Downloads SAR data from the Alaska Satellite Facility
COPYRIGHT:	(C) 2023 by NVE, Stefan Blumentrath

#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
"""

# %Module
# % description: Geocode Sentinel-1 data using the GDAR library
# % keyword: imagery
# % keyword: satellite
# % keyword: download
# % keyword: SAR
# % keyword: Sentinel
# %end

# %option G_OPT_F_INPUT
# %key: input
# % description: Comma separated list of paths to Sentinel-1 SAFE files or single text file with list of Sentinel-1 SAFE files (one path per row)
# % label: Path to input file(s)
# %end

# %option G_OPT_R_ELEV
# %end

# %option G_OPT_M_DIR
# % key: output_directory
# % required: no
# % description: Name for output directory where to store geocoded Sentinel-1 data (default: ./)
# % label: Directory where to store geocoded Sentinel-1 data
# %end

# %option G_OPT_F_INPUT
# %key: aoi
# % description: Path to GeoJSON file with the Area Of Interest (aoi)
# % label: Path to GeoJSON file with the Area Of Interest (aoi)
# %end

# %option G_OPT_F_INPUT
# %key: register_file
# % required: no
# % description: Path to register file for registering results in TGIS
# %end

# %option
# % key: polarization
# % type: string
# % required: yes
# % multiple: yes
# % description: Sentinel-1 polarization to geocode
# % label: Currently only VV and VH from Sentinel-1 GRD IW are supported
# % options: VV,VH
# % answer: VH
# %end

# %option
# % key: mode
# % required: yes
# % type: string
# % description: Sentinel-1 mode to geocode
# % label: Currently only IW mode is supported for Sentinel-1
# % options: IW
# % answer: IW
# %end

# %option
# % key: scale
# % required: yes
# % type: string
# % description: Scaling applied to output values
# % label: Currently only dbi scaling is supported
# % multiple: no
# % options: abs,intensity,dba,dbi
# % answer: dbi
# %end

# %option
# % key: suffix
# % required: yes
# % type: string
# % description: Suffix added to geocoded file / map names
# % answer: gec
# %end

# % option G_OPT_M_NPROCS
# %end

# %flag
# % key: a
# % description: Apply precision orbit information if available
# %end

# Todo:
# - harmonize with i.sentinel1.pyrosargeocode
# - read geojson to geometry string and pass that to get_target_geometry

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path


import numpy as np

import grass.script as gs


def get_raster_gdalpath(map_name, check_linked=True):
    """Get get the path to a raster map that can be opened by GDAL
    Checks for GDAL source of linked raster data and returns those
    if not otherwise requested"""
    if check_linked:
        gis_env = gs.gisenv()
        map_info = gs.find_file(map_name)
        header_path = (
            Path(gis_env["GISDBASE"])
            / gis_env["LOCATION_NAME"]
            / map_info["mapset"]
            / "cell_misc"
            / map_info["name"]
            / "gdal"
        )
        if header_path.exists():
            gdal_path = Path(
                gs.parse_key_val(header_path.read_text().replace(": ", "="))["file"]
            )
            if gdal_path.exists():
                return gdal_path
        gdal_path = Path(
            gs.parse_key_val(
                gs.parse_command("r.info", flags="e", map=map_name)["comments"]
                .replace("\\", "")
                .replace('"', ""),
                vsep=" ",
            )["input"]
        )
        if gdal_path.exists():
            return gdal_path
    gdal_path = Path(gs.find_file(map_name)["file"].replace("/cell/", "/cellhd/"))
    if gdal_path.exists():
        return gdal_path
    gs.fatal(_("Cannot determine GDAL readable path to raster map {}").format(map_name))


def check_file_input(file_input):
    """Checks input for files to geocode.
    If input is a directory (and not SAFE), contained SAFE files are listed
    If input is a text file each line is assumed to be a path to a SAFE file
    If input is a comma separated list of files element is assumed to be a path to a SAFE file
    Returns a sanetized list of Sentinel-1 input files.
    """
    file_input = options["input"].split(",")
    if len(file_input) == 1:
        file_input = Path(file_input[0])
        if file_input.is_dir() and file_input.suffix != ".SAFE":
            # Directory mode
            file_input = list(file_input.glob("*.SAFE"))
        elif file_input.suffix == ".SAFE":
            # SAFE-file mode
            file_input = check_files_list([str(file_input)])
        elif file_input.suffix.lower() == ".zip":
            gs.fatal(_("Geocoding of zipped files is currently not supported"))
        else:
            # Text file mode
            file_input = file_input.read_text(encoding="UTF8").split("\n")
            file_input = check_files_list(file_input)
    else:
        # File-list mode
        file_input = check_files_list(file_input)
    return file_input


def check_files_list(file_path_list):
    """Checks if files in a list of files exist and gives a warning otherwise"""
    existing_paths = []
    for file_path in file_path_list:
        file_path_object = Path(file_path)
        if file_path_object.exists():
            if file_path_object.suffix.upper() == ".SAFE":
                existing_paths.append(file_path_object)
            else:
                gs.warning(
                    _(
                        "Format of file {} not supported.\nOnly SAFE format is accepted."
                    ).format(file_path)
                )
        else:
            gs.warning(_("File {} not found").format(file_path))
    return existing_paths


def grass2gdar(map_name):
    """Read GRASS GIS raster map as a single band GDAR raster object"""
    gdal_path = get_raster_gdalpath(map_name)
    try:
        return reader(str(gdal_path))
    except ValueError:
        gs.fatal(
            _("Input digital elevation model {} is not a linked GeoTiff").format(
                map_name
            )
        )


def get_aoi_geometry(geojson_file):
    """Extract the Area of Interest AOI from a GeoJSON file and
    return it as an OGR Geometry object.
    The input GeoJSON should contain only one polygon geometry"""
    ogr_dataset = ogr.Open(
        f"/vsicurl/{geojson_file}" if geojson_file.startswith("http") else geojson_file
    )
    if not ogr_dataset:
        gs.fatal(_("Could not open AOI file <{}>").format(geojson_file))
    if ogr_dataset.GetLayerCount() > 1:
        gs.warning(_("Input file contains more than one layer"))
    ogr_layer = ogr_dataset.GetLayerByIndex(0)
    if ogr_layer.GetGeomType() != 3:
        gs.warning(_("GeoJSON does not contain polygons"))
    if ogr_layer.GetFeatureCount() > 1:
        gs.warning(
            _("GeoJSON contains more than one geometry. Using only the first one.")
        )
    ogr_feature = ogr_layer.GetFeature(0)
    return ogr_feature.geometry()


def get_target_geometry(bpol, geojson_file=None, crs_wkt=None):
    """Intersect a bounding polygon represented aS an array of vertices
    or an ogr.Geometry with an area of interest if given and projects
    the relevant bounding geometry (target geometry) to the target CRS
    given in WKT format."""
    # aoi = get_aoi_geometry(aoi)
    s_srs = osr.SpatialReference()
    s_srs.ImportFromEPSG(4326)
    t_srs = osr.SpatialReference()
    t_srs.ImportFromWkt(crs_wkt)
    crs_transformer = osr.CoordinateTransformation(s_srs, t_srs)
    if geojson_file:
        ogr_dataset = ogr.Open(
            f"/vsicurl/{geojson_file}"
            if geojson_file.startswith("http")
            else geojson_file
        )
        if not ogr_dataset:
            gs.fatal(_("Could not open AOI file <{}>").format(geojson_file))
        if ogr_dataset.GetLayerCount() > 1:
            gs.warning(_("Input file contains more than one layer"))
        ogr_layer = ogr_dataset.GetLayerByIndex(0)
        if ogr_layer.GetGeomType() != 3:
            gs.warning(_("GeoJSON does not contain polygons"))
        if ogr_layer.GetFeatureCount() > 1:
            gs.warning(
                _("GeoJSON contains more than one geometry. Using only the first one.")
            )
        aoi = ogr_layer.GetFeature(0)
        if not isinstance(bpol, ogr.Geometry):
            coords = ", ".join([f"{point[1]} {point[0]}" for point in list(bpol)])
            bpol = ogr.CreateGeometryFromWkt(f"POLYGON(({coords}))")
        if not bpol.Intersect(aoi.geometry()):
            return None
        bpol = bpol.Intersection(aoi.geometry())
    if isinstance(bpol, ogr.Geometry):
        return np.array(
            [
                crs_transformer.TransformPoint(
                    bpol.GetGeometryRef(0).GetY(vertex_id),
                    bpol.GetGeometryRef(0).GetX(vertex_id),
                )
                for vertex_id in range(bpol.GetGeometryRef(0).GetPointCount())
            ]
        )[:, 0:2]
    return np.array([crs_transformer.TransformPoint(*point) for point in bpol])[:, 0:2]


def gdar_geocode(
    s1_file_path,
    use_precision_orbit=True,
    module_options=None,
    output_directory=None,
):
    """Perform actual geocding"""

    mode = module_options["mode"]
    suffix = module_options["suffix"]
    polarizations = module_options["polarization"].split(",")
    out_type = module_options["scale"]

    # Read Sentinel-1 file
    s1_file = reader(str(s1_file_path))

    # Check if correct mode is selected for the given product
    if mode not in s1_file.trait_names():
        gs.fatal(
            _("File {s1_file} does not contain {mode} data").format(
                s1_file=str(s1_file_path), mode=mode
            )
        )

    # Check if correct polarization is selected for the given product
    for pol in polarizations:
        if pol in s1_file[mode]:
            gs.fatal(
                _("File {s1_file} does not contain the {pol} polarization").format(
                    s1_file=str(s1_file_path), pol=pol
                )
            )

    # Use precision orbit if requested
    if use_precision_orbit:
        gs.debug(_("Accessing precision orbits from remote..."))
        try:
            s1_file = sentinel1_orbit.update_trajectory_from_remote(s1_file)
        except ConnectionError as connection_error:
            gs.warning(
                _(
                    "Failed to update trajectory of {file_path} from ESA API. \n{error}\nProceeding anyway."
                ).format(file_path=s1_file_path, error=connection_error)
            )

    # Read DEM into GDAR raster (resulting object cannot be pickled)
    dem = grass2gdar(module_options["elevation"])[0]

    # Gety sensing time
    sensing_time = datetime.fromisoformat(
        np.datetime_as_string(
            s1_file[mode][polarizations[0]].meta.grid.refsys.utc, unit="ms"
        )
    )

    # Get track informarion
    track = s1_file[mode][polarizations[0]].meta.track

    # Get track direction
    direction = (
        "descending"
        if is_descending(s1_file[mode][polarizations[0]].meta.grid)
        else "ascending"
    )

    # Get footprint within AOI in current location CRS
    bounding_polygon = get_target_geometry(
        s1_file.get_boundingpolygon(),
        geojson_file=module_options["aoi"],
        crs_wkt=dem.meta.grid.refsys.wkt,
    )

    # Set region aligned to DEM
    region = gs.parse_command(
        "g.region",
        flags="ug",
        align=Path(module_options["elevation"]).stem,
        n=max(bounding_polygon[:, 1]),
        s=min(bounding_polygon[:, 1]),
        e=max(bounding_polygon[:, 0]),
        w=min(bounding_polygon[:, 0]),
    )

    # Create target grid
    # from gdar.gridtools import grid_from_bpol
    # https://github.com/NVE/satskred_dev/blob/5ff9e47f41da2796d91e9af0744e4cf34caa5c58/satskred/processors/woodpecker/util.py#L292
    grid = build_crsgrid(
        {
            "origin": [
                float(region["s"]) - (float(region["nsres"]) / 2.0),
                float(region["w"]) - float(region["ewres"]) / 2.0,
            ],
            "shape": [int(region["rows"]) + 1, int(region["cols"]) + 1],
            "offset": [0, 0],
            "samplespacing": [float(region["nsres"]), float(region["ewres"])],
            "wkt": dem.meta.grid.refsys.wkt,
        }
    )

    # Resampling (up or down) dependent on DEM resolution (currently not implemented)
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/Sentinel-1-sar/resolutions/level-1-ground-range-detected
    # if float(region["nsres"]) < 20.0 or float(region["ewres"]) < 20.0:
    s1_file = rastertools.downsample(s1_file, (3, 3), average=True)
    # else:
    #    s1_file = rastertools.upsample(s1_file, (3, 3), average=True)

    # Geocode
    gec = geocoding(s1_file, dem, order=3, out_type=out_type, grid=grid)

    #
    register_strings = []
    for polarization in polarizations:
        map_name = f"{s1_file_path.stem}_{suffix}_{polarization}_{out_type}_{track}_{direction}"
        output_file = output_directory / f"{map_name}.tif"
        # Write file
        write_crs(gec[mode][polarization], str(output_file))
        # Link resulting GeoTiff (may be empty)
        try:
            gs.run_command("r.external", flags="om", input=output_file, output=map_name)
            register_strings.append(
                f"{map_name}|{sensing_time}|s1_{polarization}_backscatter_{out_type}_{track}_{direction}"
            )
        except Exception:
            gs.warning(
                _(
                    "No valid pixels after geocoding {file_name}.\nPlease check the area of interest or DEM."
                ).format(file_name=str(s1_file_path))
            )
            return None

    return "\n".join(register_strings)


def main():
    """Do the main work"""
    # Inputs
    # output_dir
    output_directory = Path(options["output_directory"])
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        gs.fatal(
            _("Output directory {} is not accessibe").format(
                options["output_directory"]
            )
        )

    # identify Sentinel-1 SAFE files to geocode
    file_input = check_file_input(options["input"])

    # Check if files to geocode are found
    if len(file_input) < 1:
        gs.fatal(_("No Sentinel-1 files found to Geocode"))

    # Geocode files (in parallel if requested)
    # Setup function
    _geocode = partial(
        gdar_geocode,
        use_precision_orbit=flags["a"],
        module_options=options,
        output_directory=output_directory,
    )
    nprocs = int(options["nprocs"])
    if nprocs > 1:
        with Pool(min(nprocs, len(file_input))) as pool:
            geocoded_files = pool.map(_geocode, file_input)
    else:
        geocoded_files = []
        for s1_file in file_input:
            geocoded_files.append(_geocode(s1_file))

    # Filter out empty results and merge
    geocoded_files = (
        "\n".join(
            [register_string for register_string in geocoded_files if register_string]
        )
        + "\n"
    )
    if options["register_file"]:
        # Write registration files
        Path(options["register_file"]).write_text(geocoded_files, encoding="UTF8")
    else:
        print(geocoded_files)


if __name__ == "__main__":
    options, flags = gs.parser()

    try:
        from gdar import rastertools
        from gdar.fileformats import write_crs
        from gdar.gridtools import is_descending, build_crsgrid
        from gdar.readers import reader, sentinel1_orbit
        from gdargeocoding.geocode import geocoding
    except ImportError:
        gs.fatal(
            _(
                "Can not import gdar library. Please install it from your local repository and make sure it is on PATH"
            )
        )

    try:
        from osgeo import ogr, osr

    except ImportError:
        gs.fatal(
            _(
                "Can not import GDAL python bindings. Please install it with 'pip install GDAL==${GDAL_VERSION}'"
            )
        )

    main()
