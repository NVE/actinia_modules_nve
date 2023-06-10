#!/usr/bin/env python3

############################################################################
#
# MODULE:       t.rast.stats
# AUTHOR(S):    Stefan Blumentrath
#
# PURPOSE:      Compute area statistics of maps in a SpaceTimeRasterDataset
# COPYRIGHT:    (C) 2023 by the Stefan Blumentrath and
#               the GRASS GIS Development Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#############################################################################

# %module
# % description: Compute area statistics of maps in a SpaceTimeRasterDataset
# % keyword: temporal
# % keyword: statistics
# % keyword: raster
# % keyword: time
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_R_INPUT
# % key: zone
# % multiple: yes
# % required: no
# %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % required: no
# % multiple: no
# %end

# %option G_OPT_F_SEP
# % guisection: Format
# %end

# %option
# % key: null_value
# % type: string
# % required: no
# % multiple: no
# % key_desc: string
# % description: String representing NULL value
# % answer: *
# % guisection: Format
# %end

# %option
# % key: nsteps
# % type: integer
# % required: no
# % multiple: no
# % description: Number of floating-point subranges to collect stats from
# % answer: 255
# % guisection: Format
# %end

# %option
# % key: sort
# % type: string
# % required: no
# % multiple: no
# % options: asc,desc
# % label: Sort output statistics by cell counts
# % description: Default: sorted by categories or intervals
# % descriptions: asc; Sort by cell counts in ascending order, desc; Sort by cell counts in descending order
# % guisection: Format
# %end

# %option
# % key: region_relation
# % description: Process only maps with this spatial relation to the current computational region
# % options: overlaps,contains,is_contained
# % required: no
# % multiple: no
# %end

# %option G_OPT_M_NPROCS
# % key: nprocs
# % type: integer
# % description: Number of r.stats processes to run in parallel
# % required: no
# % multiple: no
# % answer: 1
# %end

# %flag
# % key: R
# % description: Use the raster map regions for univar statistical calculation instead of the current region
# %end

# %flag
# % key: h
# % description: Print header
# % guisection: Format
# %end

# %flag
# % key: a
# % description: Print area totals in square meters
# % guisection: Format
# %end

# %flag
# % key: c
# % description: Print cell counts (sortable)
# % guisection: Format
# %end

# %flag
# % key: p
# % description: Print approximate (total percent may not be 100%) percents
# % guisection: Format
# %end

# %flag
# % key: l
# % description: Print category labels
# % guisection: Format
# %end

# %flag
# % key: 1
# % description: One cell (range) per line
# % guisection: Format
# %end

# %flag
# % key: g
# % description: Print grid coordinates (east and north)
# % guisection: Format
# %end

# %flag
# % key: x
# % label: Print x and y (column and row)
# % description: Indexing starts with 1: first column and row are 1
# % guisection: Format
# %end

# %flag
# % key: A
# % description: Print averaged values instead of intervals (floating-point mapsonly)
# % guisection: Format
# %end

# %flag
# % key: r
# % description: Print raw indexes of floating-point ranges (floating-point mapsonly)
# % guisection: Format
# %end

# %flag
# % key: C
# % description: Report for cats floating-point ranges (floating-point maps only)
# % guisection: Format
# %end

# %flag
# % key: i
# % description: Read floating-point map as integer (use map's quant rules)
# % guisection: Format
# %end

# %flag
# % key: n
# % description: Do not report no data value
# % guisection: Format
# %end

# %flag
# % key: N
# % description: Do not report cells where all maps have no data
# % guisection: Format
# %end

# %rules
# % exclusive: -A, -C, -i, -r
# % exclusive: -a, -c, -p
# % exclusive: -x, -g
# % requires: -x, -1
# % requires: -g, -1
# %end

# ToDo:
# Implement filtering by computational region

import sys

from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE
from functools import partial

import grass.script as gs
from grass.pygrass.modules import Module


def compute_statistics(
    stats_module,
    input_tuple,
    use_map_region=False,
    separator="|",
):
    """Run the pygrass r.stats modules with input and return stdout
    :param stats_module: A PyGRASS Module object with a pre-configured
                           r.stats module
    :param input_tuple: A tuple containg the full map name, start-time,
                        end-time and semantic_label of the map

    :return: string with stdout from r.stats
    """

    if stats_module.inputs.input:
        stats_module.inputs.input = ",".join(
            (*stats_module.inputs.input, input_tuple[0])
        )
    else:
        stats_module.inputs.input = input_tuple[0]
    if use_map_region:
        stats_module.env = gs.region_env(raster=input_tuple[0], align=input_tuple[0])
    stats_module.run()

    if not stats_module.outputs.stdout:
        return None

    join_string = f"{input_tuple[0]}{separator}{input_tuple[1]}{separator}{input_tuple[2]}{separator}"

    return (
        f"{join_string}"
        + f"\n{join_string}".join(
            stats_module.outputs.stdout.rstrip().split("\n")
        ).lstrip()
    )


def compute_statistics_of_temporal_map(
    map_list,
    stats_module,
    flags,
    separator="|",
    nprocs=1,
):
    """Compute area statistics for a list of raster input maps with r.stats
    This is mainly a wrapper to parallelize the compute_statistics function
    :param map_list: A list of RasterDataset objects that contain the raster
                     maps that should be reclassified
    :param stats_module: A PyGRASS Module object with a pre-configured
                         r.stats module
    :param nprocs: The number of processes used for parallel computation
    :return: A list of strings with area statistics
    """
    # Choose statistics mode
    compute_statistics_partial = partial(
        compute_statistics, use_map_region=flags["R"], separator=separator
    )

    effective_nprocs = min(nprocs, len(map_list))
    if effective_nprocs > 1:
        with Pool(effective_nprocs) as p:
            output_list = p.starmap(
                compute_statistics_partial,
                [
                    (
                        deepcopy(stats_module),
                        (
                            raster_map.get_id(),
                            *raster_map.get_temporal_extent_as_tuple(),
                        ),
                    )
                    for raster_map in map_list
                ],
            )
    else:
        output_list = [
            compute_statistics(
                deepcopy(stats_module),
                (
                    raster_map.get_id(),
                    *raster_map.get_temporal_extent_as_tuple(),
                ),
            )
            for raster_map in map_list
        ]
    return output_list


def main():
    # Get the options
    input = options["input"]
    zone = options["zone"].split(",")
    where = options["where"]
    sep = gs.utils.separator(options["separator"])
    output = Path(options["output"]) if options["output"] else None
    nprocs = int(options["nprocs"])
    region_relation = options["region_relation"]

    if output:
        # CHeck if output exists and can be overwritten
        if output.exists() and not gs.overwrite():
            gs.fatal(_("Output file <{}> exists").format(str(output)))
        # Check if output file can be written
        try:
            output.write_text("")
        except OSError as e:
            gs.fatal(_("Cannot write output file <{}>").format(str(output)))

    # Initialize TGIS
    tgis.init()

    # Connect to TGIS DB
    dbif = tgis.SQLDatabaseInterfaceConnection()
    dbif.connect()

    # Open input STRDS
    sp = tgis.open_old_stds(input, "strds", dbif)

    # Get a list for selected raster maps from input STRDS
    if region_relation and gs.version():
        pass
    else:
        map_list = sp.get_registered_maps_as_objects(
            where=where, order="start_time", dbif=dbif
        )

    # Check if raster maps are selected
    if not map_list:
        dbif.close()
        gs.warning(_("Space time raster dataset <{}> is empty".format(input)))

    # Extract flags for r.stats
    rstats_flags = [flag for flag in flags if flag in "acpl1gxArnNCi" and flags[flag]]

    # Create Module object for r.stats that will be deep copied
    # and put into the process queue
    r_stats_module = Module(
        "r.stats",
        flags=rstats_flags,
        separator=sep,
        stdout_=PIPE,
        quiet=True,
        run_=False,
    )
    if zone:
        r_stats_module.inputs.input = zone

    # Generate header if needed
    if flags["h"]:
        header = ['map']
        if "g" in flags:
            header.append("east")
            header.append("north")
        elif "x" in flags:
            header.append("x")
            header.append("y")
        if options["zone"]:
            if len(zone) > 1:
                for idx in range(len(zone)):
                    zone_header = f"zone_{idx + 1}"
                    if "l" in flags:
                        header.append(f"{zone_header}_label")
                    header.append(zone_header)
            else:
                if "l" in flags:
                    header.append("zone_label")
                header.append("zone")

        header.append("raster_value")
        if "l" in flags:
            header.append("raster_value_label")
        if "a" in flags:
            header.append("area_m2")
        elif "c" in flags:
            header.append("cell_counts")
        elif "p" in flags:
            header.append("percent")

        header = f"{sep}".join(header)
        header = f"{header}\n"
    else:
        header = ""

    # Run r.stats modules for maplist
    output_list = compute_statistics_of_temporal_map(
        map_list,
        r_stats_module,
        flags,
        separator=sep,
        nprocs=nprocs,
    )

    # Return output
    if output_list:
        output_string = header + "\n".join(output_list)
        # Create new or overwrite existing
        if output:
            with open(output, "w") as output_file:
                output_file.write(output_string)
        else:
            print(output_string)
    else:
        gs.warning(_("Output is empty"))

    dbif.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    import grass.temporal as tgis

    main()
