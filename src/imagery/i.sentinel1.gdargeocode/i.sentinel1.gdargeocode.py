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
# % label: Path to GeoJSON file with the Area Of Interest (aoi)
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

# %option
# % key: polarization
# % required: yes
# % type: string
# % description: Sentinel-1 polarization to geocode
# % label: Currently only Sentinel-1 is supported
# % options: VV,VH
# % answer: VH
# %end

# %flag
# % key: a
# % description: Apply precision orbit information if available
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

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path


import numpy as np

import grass.script as gs


def grass2rasterio(map_name):
    """Open GRASS GIS raster map with rasterio"""
    gdal_path = gs.find_file(map_name)["file"].replace("/cell/", "/cellhd/")
    return rasterio.open(gdal_path)


def grass2gdar(map_name):
    """Build a single band GDAR raster object from a rasterio raster object"""
    band = 1
    rasterio_dataset = grass2rasterio(map_name)
    return build_crsmeta(
        rasterio_dataset.shape,
        [
            rasterio_dataset.bounds[1] + 0.5 * abs(rasterio_dataset.transform[4]),
            rasterio_dataset.bounds[0] + 0.5 * abs(rasterio_dataset.transform[0]),
        ],
        [abs(rasterio_dataset.transform[4]), abs(rasterio_dataset.transform[0])],
        (rasterio_dataset.crs).wkt,
        dtype=np.dtype(rasterio_dataset.dtypes[band - 1]),
        nodatavalue=rasterio_dataset.nodatavals[band - 1],
        form=rasterio_dataset.meta["driver"],
        desc=map_name,
        aux=None,
    )


def get_target_geometry(bpol, aoi=None, crs_wkt=None):
    """Intersect a bounding polygon represented aS an array of vertices
    or an ogr.Geometry with an area of interest if given and projects
    the relevant bounding geometry (target geometry) to the target CRS
    given in WKT format."""
    s_srs = osr.SpatialReference()
    s_srs.ImportFromEPSG(4326)
    t_srs = osr.SpatialReference()
    t_srs.ImportFromWkt(crs_wkt)
    crs_transformer = osr.CoordinateTransformation(s_srs, t_srs)
    if aoi:
        if not isinstance(bpol, ogr.Geometry):
            coords = ", ".join([f"{point[1]} {point[0]}" for point in list(bpol)])
            bpol = ogr.CreateGeometryFromWkt(f"POLYGON(({coords}))")
        if not bpol.Intersect(aoi):
            return None
        bpol = bpol.Intersection(aoi)

    if isinstance(bpol, ogr.Geometry):
        return np.array(
            [
                [
                    bpol.GetGeometryRef(0).GetY(vertex_id),
                    bpol.GetGeometryRef(0).GetX(vertex_id),
                ]
                for vertex_id in range(bpol.GetGeometryRef(0).GetPointCount())
            ]
        )

    return np.array([crs_transformer.TransformPoint(*point) for point in bpol])[:, 0:2]


def gdar_geocode(
    s1_file_path,
    dem=None,
    use_precision_orbit=True,
    aoi=None,
    out_type="dbi",
    mode="IW",
    polarizations=("VH", "VV"),
    suffix="gec",
    output_directory=None,
):
    """Perform actual geocding"""

    # Read Sentinel-1 file
    s1_file = reader(str(s1_file_path))
    if use_precision_orbit:
        gs.debug(_("Accessing precision orbits from remote..."))
        try:
            s1_file = sentinel1_orbit.update_trajectory_from_remote(s1_file)
        except ConnectionError as e:
            gs.warning(
                _(
                    "Failed to update trajectory of {file_path} from ESA API. \n{error}\nProceeding anyways."
                ).format(file_path=s1_file_path, error=e)
            )

    # Gety sensing time
    sensing_time = datetime.fromisoformat(
        np.datetime_as_string(
            s1_file[mode][polarizations[0]].meta.grid.refsys.utc, unit="ms"
        )
    )

    # Get track direction
    direction = (
        "descending"
        if is_descending(s1_file[mode][polarizations[0]].meta.grid)
        else "ascending"
    )

    # Get track info
    # track = s1_file[mode][polarization].meta.track

    # Get footprint within AOI in current location CRS
    bounding_polygon = get_target_geometry(
        s1_file.get_boundingpolygon(), aoi=aoi, crs_wkt=dem.grid.refsys.wkt
    )
    # Set region aligned to dem
    region = gs.parse_command(
        "g.region",
        flags="ug",
        align=dem.description,
        n=max(bounding_polygon[:, 1]),
        s=min(bounding_polygon[:, 1]),
        e=max(bounding_polygon[:, 0]),
        w=min(bounding_polygon[:, 0]),
    )
    # Create target grid
    grid = build_crsgrid(
        {
            "origin": [region["n"], region["w"]],
            "shape": [int(region["rows"]), int(region["cols"])],
            "offset": [0, 0],
            "samplespacing": [float(region["nsres"]), float(region["ewres"])],
            "wkt": dem.grid.refsys.wkt,
        }
    )

    # Resampling (currently not implemented)
    # if :
    #    s1_file = rastertools.downsample(s1_file, (3, 3), average=True)
    # elif :
    #     s1_file = rastertools.upsample(s1_file, (3, 3), average=True)

    # Geocode
    gec = geocoding(s1_file, dem, order=3, out_type=out_type, grid=grid)

    register_strings = []
    for polarization in polarizations:
        map_name = f"{s1_file_path.stem}_{polarization}_{suffix}"
        output_file = output_directory / f"{map_name}.tif"
        # Write file
        write_crs(gec[mode][polarization], output_file)
        # Link resulting GeoTiff
        gs.run_command("r.external", flags="om", input=output_file, output=map_name)
        register_strings.append(f"{map_name}|{sensing_time}|s1_intensity_{direction}")

    return "\n".join(register_strings)


def check_files_list(file_path_list):
    """Checks files in a list of files exist and gives a warning otherwise"""
    existing_paths = []
    for file_path in file_path_list:
        file_path_object = Path(file_path)
        if file_path_object.exists():
            existing_paths.append(file_path_object)
        gs.warning(_("File {} not found").format(file_path))

    return existing_paths


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

    # S1 files
    # identify files to geocode
    file_input = options["input"].split(",")
    if len(file_input) == 1:
        file_input = Path(file_input[0])
        if file_input.is_dir():
            # Directory mode
            file_input = list(file_input.glob("*.SAFE"))
        elif file_input.suffix == "SAFE":
            file_input = check_files_list([file_input])
        else:
            # File mode
            file_input = file_input.read_text(encoding="UTF8").split("\n")
            file_input = check_files_list(file_input)
    else:
        # File-list mode
        file_input = check_files_list(file_input)

    if len(file_input < 1):
        gs.fatal(_("No Sentinel-1 files found to Geocode"))

    # Read DEM into GDAR raster
    dem = grass2gdar(options["elevation"])

    # Geocode files (in parallel if requested)
    # Setup function
    _geocode = partial(
        gdar_geocode,
        dem=dem,
        use_precision_orbit=flags["a"],
        suffix="gec",
        mode=options["mode"],
        polarization=options["polarization"],
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

    # Write registration files
    Path(options["register_file"]).write_text(
        "\n".join(geocoded_files), encoding="UTF8"
    )


if __name__ == "__main__":
    options, flags = gs.parser()

    try:
        # from gdar import rastertools
        from gdar.fileformats import write_crs
        from gdar.gridtools import is_descending, build_crsgrid
        from gdar.metatools import build_crsmeta
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

    try:
        import rasterio

    except ImportError:
        gs.fatal(
            _(
                "Can not import rasterio library. Please install it with 'pip install rasterio'"
            )
        )

    main()
