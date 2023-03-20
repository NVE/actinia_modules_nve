#!/usr/bin/env python3

"""
 MODULE:       t.register.local
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Register raster data files from the local file system in a
               Space Time Raster Dataset (STRDS)
 COPYRIGHT:    (C) 2022 by Stefan Blumentrath

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

"""

# %module
# % description: Register files from the local file system as STRDS
# % keyword: temporal
# % keyword: import
# % keyword: raster
# % keyword: time
# % keyword: external
# % keyword: link
# % keyword: gdal
# %end

# %flag
# % key: e
# % description: Extend existing STRDS
# %end

# %flag
# % key: a
# % description: Auto-adjustment for lat/lon
# %end

# %flag
# % key: m
# % description: Read data range from metadata
# %end

# %flag
# % key: o
# % description: Override projection check (use current location's projection)
# %end

# %flag
# % key: r
# % description: Create fast link without data range
# %end

# %option G_OPT_M_DIR
# % required: yes
# % multiple: no
# %end

# %option
# % key: files
# % type: string
# % required: no
# % multiple: yes
# % key_desc: Comma separated list of files to register
# % description: Comma separated list of files to register
# %end

# %option G_OPT_STRDS_OUTPUT
# % required: yes
# % multiple: no
# %end

# %option G_OPT_F_INPUT
# % key: semantic_labels
# % required: no
# % multiple: no
# % key_desc: Input file with configuration for semantic labels ("-" = stdin)
# % description: File with mapping of band numbers, variables or subdatasets to semantic labels
# %end

# %option
# % key: long_name
# % label: Descriptive long name of the phenomenon the raster maps represent
# % type: string
# % required: no
# % multiple: no
# %end

# %option
# % key: suffix
# % label: Suffix of files to register
# % type: string
# % required: yes
# % multiple: no
# %end

# %option
# % key: time_format
# % label: datetime string format description for extracting date/time from file name (e.g. "%Y%m%d_%H%M")
# % description: Default extracts 8 numbers between underscores
# % type: string
# % required: no
# % answer: "%Y%m%d"
# % multiple: no
# %end

# %option
# % key: file_pattern
# % label: Regular expression for filtering files based on file name
# % description: Regular expression for filtering files based on file name
# % type: string
# % required: no
# % answer: *
# % multiple: no
# %end

# %option
# % key: start_time
# % label: Earliest timestamp of files to register
# % description: Timestamp in ISO format "YYYY-MM-DD HH:MM:SS"
# % type: string
# % required: no
# % multiple: no
# %end

# %option
# % key: end_time
# % label: Latest timestamp of files to register
# % description: Timestamp in ISO format "YYYY-MM-DD HH:MM:SS"
# % type: string
# % required: no
# % multiple: no
# %end

# %option G_OPT_M_COLR
# % description: Color table to assign to imported datasets
# % answer: viridis
# %end

# %option
# % key: nodata
# % type: string
# % required: no
# % multiple: no
# % key_desc: Source nodata
# % description: NoData value in source raster maps
# %end

# %option
# % key: units
# % type: string
# % required: yes
# % multiple: no
# % description: Units of the values in the input raster maps
# %end

# %option G_OPT_M_NPROCS
# % required: no
# % multiple: no
# % guisection: Settings
# %end

# %rules
# % excludes: input,files
# % required: input,files
# %end

# ToDo:
# - add rules to secure consistent user input

import os
import re
import subprocess
import sys

from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import chain
from math import inf
from multiprocessing import Pool
from pathlib import Path

import grass.script as gscript
import grass.temporal as tgis

from grass.pygrass.gis import Mapset
from grass.pygrass.modules.interface import Module, MultiModule
from grass.temporal.register import register_maps_in_space_time_dataset


def legalize_name_string(string):
    """
    Replace conflicting characters with _
    :param string: string to modify if needed
    :return: modified string
    """
    legal_string = re.sub(r"[^\w\d-]+|[^\x00-\x7F]+|[ -/\\]+", "_", string)
    return legal_string


def strftime_to_regex(string):
    """
    Transform a strftime format string to a regex for extracting time from file name
    :param string: strftime format string
    :return: regex string
    """
    digits = {
        "%Y": "[0-9]{4}",
        "%m": "[0-9]{2}",
        "%d": "[0-9]{2}",
        "%M": "[0-9]{2}",
        "%H": "[0-9]{2}",
        "%S": "[0-9]{2}",
    }
    for item, digit in digits.items():
        string = string.replace(item, digit)
    if "%" in string:
        gscript.warning(_("Unmatched format item in input string"))
    return string


def parse_semantic_label_conf(conf_file, grass_major_version):
    """
    Read user provided mapping of subdatasets / variables to semantic labels
    Return a dict with mapping, bands or subdatasets that are not mapped in this file are skipped
    from import
    ToDo: move to pygrass / grass core
    :param conf_file:
    :param grass_major_version:
    :return: Dictionary with semantic labels
    """
    if conf_file is None or conf_file == "":
        return None

    if grass_major_version < 8:
        gscript.warning(
            _(
                "The semantic labels concept requires GRASS GIS version 8.0 or later.\n"
                "Semantic labels will not be written but used for filtering input data.".format(
                    conf_file=conf_file
                )
            )
        )
    else:
        # Lazy import GRASS GIS 8 function if needed
        from grass.lib.raster import Rast_legal_semantic_label

    semantic_label = {}
    if not os.access(conf_file, os.R_OK):
        gscript.fatal(
            _(
                "Cannot read configuration file <{conf_file}>".format(
                    conf_file=conf_file
                )
            )
        )

    with open(conf_file, "r") as c_file:
        configuration = c_file.read()
        for idx, line in enumerate(configuration.split("\n")):
            if line.startswith("#") or line == "":
                continue
            if len(line.split("=")) == 2:
                line = line.split("=")
                # Check if assigned semantic label has legal a name
                if grass_major_version < 8 or Rast_legal_semantic_label(line[1]) == 1:
                    semantic_label[line[0]] = line[1]
                else:
                    gscript.fatal(
                        _(
                            "Line {line_nr} in configuration file <{conf_file}> "
                            "contains an illegal band name".format(
                                line_nr=idx + 1, conf_file=conf_file
                            )
                        )
                    )
            else:
                gscript.fatal(
                    _(
                        "Invalid format of semantic label configuration in file <{}>".format(
                            conf_file
                        )
                    )
                )

    return semantic_label


def parse_geotransform(geotransform_string):
    """This is part of a hack for netCDF data at NVE
    with invalid GeoTranfsorm information"""
    return list(
        map(
            float,
            geotransform_string.replace("{", "")
            .replace("}", "")
            .replace(" ", "")
            .split(","),
        )
    )


def map_semantic_labels(
    raster_dataset, semantic_label_dict=None, input_option_dict=None
):
    """
    Map semantic labels from config file to band(s) or subdataset(s) / variable(s) in input raster maps
    :param raster_dataset:
    :param semantic_labels:
    :return: List of tuples
    """
    raster_dataset = str(raster_dataset)
    ds = gdal.Open(raster_dataset)
    if not ds:
        gscript.warning(_("Cannot open dataset <{}>".format(raster_dataset)))
        print("Invalid input")
        return []
    # Map subdatasets if present
    subdatasets = ds.GetSubDatasets()
    nc_metadata = ds.GetMetadata()
    geotransforms = [
        key for key in ds.GetMetadata().keys() if "geotransform" in key.lower()
    ]
    if subdatasets:
        if semantic_label_dict and not any(
            [
                reference[1].split(" ")[1] in semantic_label_dict
                for reference in subdatasets
            ]
        ):
            gscript.warning(_("No subdatasets to import."))
            return []
        elif semantic_label_dict:
            import_tuple = []
            for subdataset in subdatasets:
                # This is a hack for netCDF data at NVE with invalid GeoTranfsorm information
                sds = subdataset[1].split(" ")[1]
                if sds not in semantic_label_dict:
                    continue
                if len(geotransforms) > 1:
                    if len(sds) >= 2:
                        geotransform = [key for key in geotransforms if sds[1] in key]
                        if not geotransform:
                            continue
                    geotransform = parse_geotransform(nc_metadata[geotransform[0]])
                elif len(geotransforms) == 1:
                    geotransform = parse_geotransform(nc_metadata[geotransforms[0]])
                else:
                    geotransform = None

                import_tuple.append(
                    (
                        create_vrt(
                            subdataset[0],
                            input_option_dict,
                            input_option_dict["nodata"],
                            geotransform,
                        ),
                        1,
                        semantic_label_dict[sds],
                    )
                )

        else:
            import_tuple = []
            for subdataset in subdatasets:
                # This is a hack for netCDF data at NVE with invalid GeoTranfsorm information
                sds = subdataset[1].split(" ")[1]
                if sds not in semantic_label_dict:
                    continue
                if len(geotransforms) > 1:
                    if len(sds) >= 2:
                        geotransform = [key for key in geotransforms if sds[1] in key]
                        if not geotransform:
                            continue
                    geotransform = parse_geotransform(nc_metadata[geotransform[0]])
                elif len(geotransforms) == 1:
                    geotransform = parse_geotransform(nc_metadata[geotransforms[0]])
                else:
                    geotransform = None

                import_tuple.append(
                    (
                        create_vrt(
                            subdataset[0],
                            input_option_dict,
                            input_option_dict["nodata"],
                            geotransform,
                        ),
                        1,
                        legalize_name_string(sds),
                    )
                )

    # This is a hack for netCDF data at NVE with invalid GeoTranfsorm information
    elif not subdatasets and raster_dataset.endswith("nc") and geotransforms:
        geotransform = parse_geotransform(nc_metadata[geotransforms[0]])
        if semantic_label_dict:
            import_tuple = [
                (
                    create_vrt(
                        raster_dataset,
                        input_option_dict,
                        input_option_dict["nodata"],
                        geotransform,
                    ),
                    band,
                    semantic_label_dict[str(band)],
                )
                for band in range(1, ds.RasterCount + 1)
            ]
        else:
            import_tuple = [
                (
                    create_vrt(
                        raster_dataset,
                        input_option_dict,
                        input_option_dict["nodata"],
                        geotransform,
                    ),
                    band,
                    str(band),
                )
                for band in range(1, ds.RasterCount + 1)
            ]
    # Map bands
    else:
        if semantic_label_dict:
            import_tuple = [
                (raster_dataset, band, semantic_label_dict[str(band)])
                for band in range(1, ds.RasterCount + 1)
            ]
        else:
            import_tuple = [
                (raster_dataset, band, str(band))
                for band in range(1, ds.RasterCount + 1)
            ]
    ds = None
    return import_tuple


def create_vrt(subdataset, gisenv, nodata, geotransform, recreate=False):
    """
    Create a GDAL VRT for import
    Assumes units and long_name are valid for all subdatasets

    :param subdataset:
    :param gisenv:
    :param nodata:
    :param recreate:
    :return:
    """
    vrt_dir = Path(gisenv["GISDBASE"]).joinpath(
        gisenv["LOCATION_NAME"], gisenv["MAPSET"], "gdal"
    )
    sds = subdataset.split(":")
    if len(sds) >= 3:
        sds_path = sds[1]
        sds_name = sds[2]
        vrt = vrt_dir.joinpath(
            "gdal_{}.vrt".format(
                legalize_name_string(f"{Path(sds_path).stem}_{sds[2]}")
            )
        )
    else:
        vrt = vrt_dir.joinpath(
            "gdal_{}.vrt".format(legalize_name_string(f"{Path(subdataset).stem}"))
        )
    vrt_name = str(vrt)
    if vrt.exists() and not recreate:
        return vrt_name
    kwargs = {"format": "VRT"}
    if nodata is not None:
        kwargs["noData"] = nodata
    vrt = gdal.Translate(
        vrt_name,
        subdataset,
        options=gdal.TranslateOptions(**kwargs),
    )
    if geotransform:
        vrt.SetGeoTransform(geotransform)
    vrt = None
    vrt = vrt_name

    return vrt


def import_data(import_tuple, metadata_dict=None, modules_dict=None):
    """
    Link (import) external raster data and set relevant metadata
    Implemented as a sequence of modules to run
    Returns a line for registering the linked map in a STRDS
    :param import_tuple: tuple containing Path to the raster data file to link/import, band/subdataset, semantic_label
    :param metadata_dict: dictionary containing relevant metadata for the link/import
    :param modules_dict: Dictionary containing pyGRASS modules to run for link/import
    :return: str with one line for registering the linked raster map in an STRDS
    :rtype: str
    """
    file_path = Path(import_tuple[0])
    date_pattern = f".*({strftime_to_regex(metadata_dict['time_format'])}).*"
    time_string = re.match(date_pattern, str(file_path)).groups()[0]
    time_stamp = datetime.strptime(time_string, metadata_dict["time_format"])
    time_stamp_iso = time_stamp.isoformat(sep=' ')
    suffix = f"_{import_tuple[2]}" if import_tuple[2] else ""
    output_name = (
        f"map_{file_path.stem}{suffix}"
        if file_path.stem[0].isdigit()
        else f"{file_path.stem}{suffix}"
    )
    mods = deepcopy(modules_dict)
    mods["import"].inputs.input = str(file_path)
    mods["import"].inputs.band = import_tuple[1]
    mods["import"].outputs.output = output_name
    mods["metadata"].inputs.map = output_name
    mods["metadata"].inputs.units = metadata_dict["units"]
    mods["metadata"].inputs.title = metadata_dict["long_name"]
    mods["metadata"].inputs.semantic_label = import_tuple[2]
    mods["timestamp"].inputs.map = output_name
    mods["timestamp"].inputs.date = time_stamp.strftime("%-d %b %Y")
    try:
        MultiModule(list(mods.values())).run()
    except:
        gscript.warning(_("Cannot register file <{}>".format(import_tuple[0])))
        return None

    return f"{output_name},{time_stamp_iso},{time_stamp_iso},{import_tuple[2]}"


def main():
    """run the main workflow"""
    # Get the current mapset
    mapset = Mapset()

    # Get number of cores to use for processing
    nprocs = int(options["nprocs"])

    # Add gisenv to options
    options.update(gscript.gisenv())

    # Get minimum start time
    start_time = options["start_time"]
    if start_time:
        try:
            start_time = datetime.fromisoformat(options["start_time"])
        except ValueError:
            gscript.fatal(
                _(
                    "Cannot parse timestamp provided in <start_time> option."
                    "Please make sure it is provided in ISO format (YYYY-MM-DD HH:MM:SS)"
                )
            )

    # Get maximum start time
    end_time = options["end_time"]
    if end_time:
        try:
            end_time = datetime.fromisoformat(options["end_time"])
        except ValueError:
            gscript.fatal(
                _(
                    "Cannot parse timestamp provided in <end_time> option."
                    "Please make sure it is provided in ISO format (YYYY-MM-DD HH:MM:SS)"
                )
            )


    if options["semantic_labels"]:
        semantic_labels = parse_semantic_label_conf(options["semantic_labels"], 8)
    else:
        semantic_labels = None

    # Identify files to import
    if options["files"]:
        raster_files = [
            Path(options["input"]) / raster_file
            for raster_file in options["files"].split(",")
            if Path(options["input"]).joinpath(raster_file).exists()
        ]
    else:
        raster_files = Path(options["input"]).rglob(
            f"{options['file_pattern']}.{options['suffix']}"
        )
        end_timestamp = end_time.timestamp() if end_time else inf
        start_timestamp = start_time.timestamp() if start_time else 0.0
        if start_time:
            raster_files = [
                nc_file
                for nc_file in raster_files
                if end_timestamp >= nc_file.stat().st_mtime >= start_timestamp
            ]
        else:
            raster_files = list(raster_files)

    # Abort if no files to import are found
    if not raster_files:
        gscript.fatal(_("No files found to import."))

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{mapset.name}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if tgis_strds.is_in_db() and not gscript.overwrite():
        gscript.fatal(
            _(
                "Output STRDS <{}> exists."
                "Use --overwrite with or without -e to modify the existing STRDS."
            ).format(options["output"])
        )
    if not tgis_strds.is_in_db() or (gscript.overwrite() and not flags["e"]):
        Module(
            "t.create",
            output=options["output"],
            type="strds",
            temporaltype="absolute",
            title=options["long_name"],
            description=options["long_name"],
            verbose=True,
        )

    # Create initial module objects for import of raster data
    modules = {
        "import": Module(
            "r.external",
            quiet=True,
            run_=False,
            flags="".join([flag for flag in "amro" if flags[flag]]),
            title=options["long_name"],
        ),
        "metadata": Module(
            "r.support",
            quiet=True,
            run_=False,
            title=options["long_name"],
            units=options["units"],
        ),
        "timestamp": Module(
            "r.timestamp",
            quiet=True,
            run_=False,
        ),
    }

    # Pre-define kwargs for import-function
    match_semantic_labels = partial(
        map_semantic_labels,
        semantic_label_dict=semantic_labels,
        input_option_dict=options,
    )

    # Pre-define kwargs for import-function
    run_import = partial(import_data, metadata_dict=options, modules_dict=modules)

    # Get GRASS GIS environment info
    options.update(dict(gscript.gisenv()))

    # Create directory for vrt files if needed
    vrt_dir = Path(options["GISDBASE"]).joinpath(
        options["LOCATION_NAME"], options["MAPSET"], "gdal"
    )
    if not vrt_dir.is_dir():
        vrt_dir.mkdir()

    print("Files filtered")
    print(raster_files)
    # Match semantic labels and create VRTs if needed
    with Pool(processes=nprocs) as pool:
        import_tuples = pool.map(match_semantic_labels, raster_files)

    print("Semantic labels matched")
    # Import / Link raster maps into mapset
    with Pool(processes=nprocs) as pool:
        register_string = pool.map(run_import, chain(*import_tuples))

    print("Maps imported")
    # Register imported maps in STRDS using register file
    map_file = gscript.tempfile()
    with open(map_file, "w") as m_f:
        m_f.write("\n".join(set([r_s for r_s in register_string if r_s is not None])))
    print(len(register_string))
    print(len(set([r_s for r_s in register_string if r_s is not None])))
    register_maps_in_space_time_dataset(
        "raster",
        strds_long_name,
        file=map_file,
        update_cmd_list=False,
        fs=",",
    )

    print("Maps registered")

    # Update metadata of target STRDS from newly imported maps
    # tgis_strds.update_from_registered_maps(dbif=None)


if __name__ == "__main__":
    options, flags = gscript.parser()

    # lazy imports
    try:
        from osgeo import gdal
    except ImportError:
        gscript.fatal(
            _(
                "Unable to load GDAL Python bindings (requires "
                "package 'python-gdal' or Python library GDAL "
                "to be installed)."
            )
        )

    sys.exit(main())
