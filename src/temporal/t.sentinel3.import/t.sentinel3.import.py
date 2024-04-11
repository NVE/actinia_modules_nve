#!/usr/bin/env python3

"""
 MODULE:      t.sentinel3.import
 AUTHOR(S):   Stefan Blumentrath
 PURPOSE:     Import and pre-process Sentinel-3 data from the Copernicus program
              into a Space Time Raster Dataset (STRDS)
 COPYRIGHT:   (C) 2024 by Norwegian Water and Energy Directorate, Stefan Blumentrath,
              and the GRASS development team

              This program is free software under the GNU General Public
              License (>=v2). Read the file COPYING that comes with GRASS
              for details.
"""

# %Module
# % description: Import and pre-process Sentinel-3 data from the Copernicus program into a Space Time Raster Dataset (STRDS)
# % keyword: imagery
# % keyword: satellite
# % keyword: Sentinel
# % keyword: temporal
# % keyword: import
# % keyword: optical
# % keyword: thermal
# %end

# %option
# % key: input
# % label: Sentinel-3 input data
# % description: Either a (comma separated list of) path(s) to Sentinel-3 zip files or a textfile with such paths (one per row)
# % required: yes
# % multiple: yes
# %end

# %option G_OPT_STRDS_OUTPUT
# % required: yes
# % key: output
# % multiple: no
# % description: Name of the output space time raster dataset (STRDS)
# %end

# %option
# % key: title
# % multiple: no
# % description: Title of the resulting STRDS
# % required: no
# %end

# %option
# % key: description
# % multiple: no
# % description: Description of the resulting STRDS
# % required: no
# %end

# %option
# % key: product_type
# % multiple: no
# % options: S3SL1RBT,S3SL2LST
# % answer: S3SL2LST
# % description: Sentinel-3 product type to import (currently, only S3SL1RBT and S3SL2LST are supported)
# % required: yes
# %end

# %option
# % key: bands
# % multiple: yes
# % answer: all
# % required: yes
# % description: Data bands to import (e.g. LST, default is all available)
# %end

# %option
# % key: anxillary_bands
# % multiple: yes
# % answer: all
# % required: no
# % description: Anxillary data bands to import (e.g. LST_uncertainty, default is all available)
# %end

# %option
# % key: flag_bands
# % multiple: yes
# % answer: all
# % required: no
# % description: Quality flag bands to import (e.g. bayes_in, default is all available)
# %end

# %option
# % key: maximum_solar_angle
# % type: double
# % description: Import only pixels where solar angle is lower or equal to the given maximum
# % required: no
# %end

# %option
# % key: basename
# % description: Basename used as prefix for map names (default is derived from the input file(s))
# % required: no
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# % key: c
# % description: Import LST in degree celsius (default is kelvin)
# % guisection: Settings
# %end

# %flag
# % key: d
# % description: Import data with decimals as double precision
# % guisection: Settings
# %end

# %flag
# % key: j
# % description: Write metadata json for each band to LOCATION/MAPSET/cell_misc/BAND/description.json
# % guisection: Settings
# %end

# %flag
# % key: k
# % description: Keep original cell values during interpolation (see: r.fill.stats)
# % guisection: Settings
# %end

# %flag
# % key: o
# % description: Process oblique view (default is nadir)
# % guisection: Settings
# %end

# %flag
# % key: n
# % description: Import data at native resolution of the bands (default is use current region)
# % guisection: Settings
# %end

# # %flag
# # % key: p
# # % description: Print raster data to be imported and exit
# # % guisection: Print
# # %end

# %flag
# % key: r
# % description: Rescale radiance bands to reflectance
# % guisection: Settings
# %end

# %rules
# % collective: title,description
# % required: -e,title,description
# %end


import re
import sys

from datetime import datetime
from functools import partial
from math import floor
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs


S3_FILE_PATTERN = {
    # "S3OL1ERF": None,
    "S3SL1RBT": "S3*SL_1_RBT__*.zip",
    "S3SL2LST": "S3*SL_2_LST__*.zip",
}


def check_file_input(file_input, product_type):
    """Checks input for files to geocode.
    If input is a directory (and not aSAFE), contained SAFE files are listed
    If input is a text file each line is assumed to be a path to a SAFE file
    If input is a comma separated list of files element is assumed to be a path to a SAFE file
    Returns a sanetized list of Sentinel-1 input files.
    """
    file_input = file_input.split(",")
    file_input = [Path(file_path) for file_path in file_input]
    if len(file_input) == 1:
        if file_input[0].is_dir():
            # Directory mode
            file_input = list(file_input[0].glob(S3_FILE_PATTERN[product_type]))
        elif file_input[0].suffix.lower() != ".zip":
            # Text file mode
            file_input = [
                Path(file_path)
                for file_path in file_input[0].read_text(encoding="UTF8").split("\n")
            ]
    # File-list mode
    return check_files_list(file_input, product_type)


def check_files_list(file_path_list, product_type):
    """Checks if files in a list of files exist and gives a warning otherwise"""
    if not file_path_list:
        gs.fatal(_("No scenes found to process"))
    existing_paths = []
    file_pattern = re.compile(
        f".*{S3_FILE_PATTERN[product_type].replace('*', '.*')}", re.IGNORECASE
    )
    for file_path in file_path_list:
        file_path_object = Path(file_path)
        if file_path_object.exists():
            # Match against: S3_FILE_PATTERN[product_type]
            if file_pattern.search(str(file_path_object)):
                existing_paths.append(file_path_object)
            else:
                gs.warning(
                    _(
                        "File {file_path} does not match expected pattern {pattern} of product {product_type}."
                    ).format(
                        file_path=file_path,
                        pattern=S3_FILE_PATTERN[product_type],
                        product_type=product_type,
                    )
                )
        else:
            gs.warning(_("File {} not found").format(file_path))
    return existing_paths


def parse_s3_file_name(file_name):
    """Extract info from Sentinel-3 file name according to naming convention:
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-3-slstr/naming-convention
    Assumes that file name is checked to be a valid / supported Sentinel-3 file name, e.g.:
    "S3B_SL_1_RBT____20240129T110139_20240129T110439_20240130T114811_0180_089_094_1800_PS2_O_NT_004.SEN3"
    The suffix does not matter
    :param file_name: string representing the file name of a Senintel-3 scene
    """
    try:
        return {
            "mission_id": file_name[0:3],
            "instrument": file_name[4:6],
            "level": file_name[7],
            "product": file_name[9:12],
            "start_time": datetime.strptime(file_name[16:31], "%Y%m%dT%H%M%S"),
            "end_time": datetime.strptime(file_name[32:47], "%Y%m%dT%H%M%S"),
            "ingestion_time": datetime.strptime(file_name[48:63], "%Y%m%dT%H%M%S"),
            "duration": file_name[64:68],
            "cycle": file_name[69:72],
            "relative_orbit": file_name[73:76],
            "frame": file_name[77:81],
        }
    except ValueError:
        gs.fatal(_("{} is not a supported Sentinel-3 scene").format(file_name))


def group_scenes(
    s3_files,
    group_variables=(
        "mission_id",
        "instrument",
        "level",
        "product",
        "cycle",
        "relative_orbit",
    ),
):
    """
    Group scenes along track and date by information from the file name:
     1. mission ID
     2. product type
     3. temporal granule
     4. duration
     5. cycle
     6. relative orbit
     : param s3_filesv: list of pathlib.Path objects with Sentinel-3 files
    """
    groups = {}
    for sfile in s3_files:
        s3_name_dict = parse_s3_file_name(sfile.name)
        s3_name_dict["level"] = (
            "2" if s3_name_dict["level"] == "1" else s3_name_dict["level"]
        )
        group_id = "_".join(
            [s3_name_dict[group_var] for group_var in group_variables]
            + [s3_name_dict["start_time"].strftime("%Y%m%d")]
        )
        if group_id in groups:
            groups[group_id].append(str(sfile))
        else:
            groups[group_id] = [str(sfile)]
    return groups


def process_scene_group(scene_group, module_options=None, module_flags=None):
    """Import a group of Sentinel3 scenes"""
    gs.verbose(_("Processing scene group {}...").format(scene_group[0]))

    i_sentinel3_import = Module(
        "i.sentinel3.import",
        input=",".join(scene_group[1]),
        stdout_=PIPE,
        **module_options,
        flags=module_flags,
        quiet=True,
    )
    return i_sentinel3_import.outputs.stdout


def distribute_cores(nprocs, groups_n):
    """Distribute cores across inner (parallel processes within
    i.sentinel3.import) and outer (parallel runs of i.sentinel3.import)
    loop of processes. At least one core is allocated to inner 
    (i.sentinel3.import) and outer (group of Sentinel-3 scenes)
    process.
    Order if returns is inner, outer."""
    return max(1, floor(nprocs / groups_n)), min(groups_n, nprocs)


def main():
    """Do the main work"""

    # Get GRASS GIS environment
    gisenv = dict(gs.gisenv())

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{gisenv['MAPSET']}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if input is complete and valid
    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if tgis_strds.is_in_db():
        if not gs.overwrite():
            gs.fatal(
                _(
                    "Output STRDS <{}> exists."
                    "Use --overwrite together with -e to modify the existing STRDS."
                ).format(options["output"])
            )
    elif not options["title"] or not options["description"]:
        gs.fatal(
            _(
                "Creation of a new STRDS <{}> requires the 'title' and 'description' option"
            ).format(strds_long_name)
        )

    # Group input scenes
    groups_to_process = group_scenes(
        check_file_input(options["input"], options["product_type"])
    )

    # Distribute cores
    nprocs_inner, nprocs_outer = distribute_cores(
        int(options["nprocs"]), len(groups_to_process)
    )

    # Setup import module, and collect flags amd options
    module_options = {
        option: options[option]
        for option in [
            "product_type",
            "bands",
            "anxillary_bands",
            "flag_bands",
            "basename",
            "maximum_solar_angle",
        ]
    }
    module_options["nprocs"] = nprocs_inner
    import_module = partial(
        process_scene_group,
        module_options=module_options,
        module_flags=[flag for flag in "cdnjkor" if flags[flag]],
    )

    if nprocs_outer > 1:
        with Pool(nprocs_outer) as pool:
            register_strings = pool.map(import_module, groups_to_process.items())
    else:
        register_strings = [
            import_module(scene_group) for scene_group in groups_to_process.items()
        ]

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{gisenv['MAPSET']}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if tgis_strds.is_in_db() and not gs.overwrite():
        gs.fatal(
            _(
                "Output STRDS <{}> exists."
                "Use --overwrite together with -e to modify the existing STRDS."
            ).format(options["output"])
        )

    # Create STRDS if needed
    if not tgis_strds.is_in_db() or (gs.overwrite() and not flags["e"]):
        tgis.open_new_stds(
            options["output"],
            "strds",
            "absolute",
            options["title"],
            options["description"],
            "mean",
            None,
            gs.overwrite(),
        )

    # Write registration file with unique lines
    tmp_file = gs.tempfile(create=False)
    Path(tmp_file).write_text("\n".join(register_strings) + "\n", encoding="UTF8")

    # Register downloaded maps in STRDS
    register_maps_in_space_time_dataset(
        "raster",
        strds_long_name,
        file=tmp_file,
        update_cmd_list=False,
        fs="|",
    )


if __name__ == "__main__":
    options, flags = gs.parser()

    # Lazy imports
    import grass.temporal as tgis
    from grass.pygrass.modules.interface import Module
    from grass.temporal.register import register_maps_in_space_time_dataset

    sys.exit(main())
