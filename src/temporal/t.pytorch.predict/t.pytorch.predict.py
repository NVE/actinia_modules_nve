#!/usr/bin/env python3

"""
 MODULE:      t.pytorch.predict
 AUTHOR(S):   Stefan Blumentrath
 PURPOSE:     Apply a pytorch model to imagery groups in a Space Time Raster Dataset
              and register results in an output STRDS
 COPYRIGHT:   (C) 2024 by Norwegian Water and Energy Directorate, Stefan Blumentrath,
              and the GRASS development team

              This program is free software under the GNU General Public
              License (>=v2). Read the file COPYING that comes with GRASS
              for details.
"""

# %Module
# % description: Apply a pytorch model to imagery groups in a Space Time Raster Dataset (STRDS)
# % keyword: raster
# % keyword: imagery
# % keyword: deep learning
# % keyword: pytorch
# % keyword: unet
# % keyword: GPU
# % keyword: predict
# %end

# %option G_OPT_STRDS_INPUT
# %end

# # %option
# # % key: region_relation
# # % type: string
# # % required: no
# # % multiple: no
# # %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_STRDS_OUTPUT
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

# %option G_OPT_F_INPUT
# % key: model
# % description: Path to input deep learning model file (.pt)
# %end

# %option G_OPT_F_INPUT
# % key: model_code
# % description: Path to input deep learning model code (.py)
# %end

# %option
# % key: tile_size
# % type: integer
# % required: no
# % multiple: yes
# % description: Number of rows and columns in tiles (rows, columns)
# %end

# %option
# % key: overlap
# % type: integer
# % required: no
# % multiple: no
# % description: Number of rows and columns of overlap in tiles
# %end

# %option G_OPT_F_INPUT
# % key: configuration
# % description: Path to JSON file with band configuration in the input deep learning model
# %end

# %option G_OPT_F_INPUT
# %key: mask_json
# % required: no
# % multiple: no
# % description: JSON file with one or more mask band or map name(s) and reclass rules for masking, e.g. {"mask_band": "1 thru 12 36 = 1", "mask_map": "0"}
# %end

# %option G_OPT_M_NPROCS
# %end

# %option
# % key: basename
# % type: string
# % required: no
# % multiple: no
# % description: Name for output raster map
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# % key: c
# % description: Use CPU as device for prediction, default is use cuda (GPU) if detected
# %end

# %flag
# %key: l
# % description: Limit output to valid range (data outside the valid range is set to valid min/max)
# %end

# %rules
# % collective: title,description
# % required: -e,title,description
# %end

import os
import json
import sys

# from datetime import datetime
from functools import partial
from math import floor
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs


TMP_NAME = gs.tempname(12)
# Get GRASS GIS environment
GISENV = dict(gs.gisenv())


def distribute_cores(nprocs, groups_n):
    """Distribute cores across inner (parallel processes within
    i.pytorch.predict) and outer (parallel runs of i.pytorch.predict)
    loop of processes. At least one core is allocated to inner
    (i.pytorch.predict) and outer (imagery group)
    process.
    Order if returns is inner, outer."""
    return max(1, floor(nprocs / groups_n)), min(groups_n, nprocs)


def process_scene_group(
    temporal_extent,
    map_list,
    basename=None,
    module_options=None,
    torch_flags=None,
):
    """Create an imagery group from semantic labels of a temporal extent and
    run a pytorch prediction on the imagery group"""
    # Get the base name
    if not basename:
        output_name = os.path.commonprefix(list(map_list.values())).rstrip("_")
    else:
        output_name = f"{basename}_{temporal_extent[0].isoformat()}_{temporal_extent[1].isoformat()}"
    # Get semantic labels
    output_bands = json.loads(
        Path(module_options["configuration"]).read_text(encoding="UTF8")
    )["output_bands"]

    gs.verbose(_("Processing group {}...").format(output_name))
    Module(
        "i.group",
        group=f"{TMP_NAME}_{output_name}",
        input=list(map_list.values()),
        quiet=True,
    )
    Module(
        "i.pytorch.predict",
        input=f"{TMP_NAME}_{output_name}",
        output=output_name,
        stdout_=PIPE,
        **module_options,
        flags=torch_flags,
        quiet=True,
    )
    register_strings = [
        "|".join(
            [
                f"{output_name}_{output_band}@{GISENV['MAPSET']}",
                temporal_extent[0].isoformat(),
                temporal_extent[1].isoformat(),
                band_dict["semantic_label"],
            ]
        )
        for output_band, band_dict in output_bands.items()
    ]
    return "\n".join(register_strings)


def main():
    """Do the main work"""

    # Initialize TGIS
    dbif = tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{GISENV['MAPSET']}"
    output_strds = tgis.SpaceTimeRasterDataset(strds_long_name)
    output_strds_in_db = output_strds.is_in_db()
    overwrite = gs.overwrite()

    # Check if input is complete and valid
    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if output_strds_in_db:
        if not overwrite:
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

    # Get list of maps in input STRDS
    input_strds = tgis.open_old_stds(options["input"], "strds", dbif)
    map_rows = input_strds.get_registered_maps(
        ",".join(["id", "start_time", "end_time", "semantic_label"]),
        options["where"],
        "start_time",
        dbif,
    )

    if not map_rows:
        gs.warning(_("No data selected from STRDS <{}>").format(options["input"]))
        sys.exit(0)

    # Group maps using granule
    map_groups = {}
    for row in map_rows:
        if (row["start_time"], row["end_time"]) not in map_groups:
            map_groups[(row["start_time"], row["end_time"])] = {
                row["semantic_label"]: row["id"]
            }
        else:
            map_groups[(row["start_time"], row["end_time"])][
                row["semantic_label"]
            ] = row["id"]

    # Distribute cores
    nprocs_inner, nprocs_outer = distribute_cores(
        int(options["nprocs"]), len(map_groups)
    )

    # Collect basic module_options for i.pytorch.predict
    module_options = {
        option: (
            list(map(int, options[option].split(",")))
            if option == "tile_size"
            else options[option]
        )
        for option in [
            "model",
            "model_code",
            "tile_size",
            "overlap",
            "configuration",
            "mask_json",
        ]
    }
    module_options["nprocs"] = nprocs_inner

    # Collect pytorch flags
    torch_flags = [flag for flag in "lc" if flags[flag]]

    # Setup prediction module function
    i_pytorch_predict = partial(
        process_scene_group,
        module_options=module_options,
        basename=options["basename"],
        torch_flags=torch_flags,
    )

    # Run predictions and collect
    if nprocs_outer > 1:
        with Pool(nprocs_outer) as pool:
            register_strings = pool.starmap(i_pytorch_predict, map_groups.items())
    else:
        register_strings = [
            i_pytorch_predict(*scene_group) for scene_group in map_groups.items()
        ]

    # Create STRDS if needed
    if not output_strds_in_db or (overwrite and not flags["e"]):
        tgis.open_new_stds(
            options["output"],
            "strds",
            "absolute",
            options["title"],
            options["description"],
            "mean",
            None,
            overwrite,
        )

    # Write registration file with unique lines
    tmp_file = gs.tempfile(create=False)
    Path(tmp_file).write_text("\n".join(register_strings) + "\n", encoding="UTF8")

    # Register results in output STRDS
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

# Check output STRDS
# Group scenes
# Import scenes
# register pre-processed scenes in output STRDS
