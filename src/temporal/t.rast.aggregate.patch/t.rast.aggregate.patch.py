#!/usr/bin/env python

"""
MODULE:    t.rast.aggregate.patch
AUTHOR(S): Stefan Blumentrath

PURPOSE:   Aggregates rasters maps in space and time by means of patching (r.patch/r.buildvrt)
COPYRIGHT: (C) 2024 by Stefan Blumentrath, Norwegian Water and Energy Directorate and the GRASS Development Team

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
# % description: Aggregate multiple space time raster maps into mosaics with the given granualrity using r.patch or r.buildvrt.
# % keyword: temporal
# % keyword: aggregation
# % keyword: patch
# % keyword: raster
# % keyword: merge
# % keyword: patching
# % keyword: granularity
# % keyword: strds
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_T_WHERE
# %end

# %option
# % key: sort
# % description: Sort order of input maps using start_time (default: desc = newest first)
# % options: asc,desc
# % answer: desc
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

# %option
# % key: basename
# % multiple: no
# % description: Basename for output raster maps
# % required: no
# %end

# %option
# % key: offset
# % type: integer
# % description: Offset that is used to create the output map ids, output map id is generated as: basename_ (count + offset)
# % required: no
# % multiple: no
# % answer: 0
# %end

# %option
# % key: suffix
# % type: string
# % description: Suffix to add at basename: set 'gran' for granularity, 'time' for the full time format, 'num' for numerical suffix with a specific number of digits (default %05)
# % answer: gran
# % required: no
# % multiple: no
# %end

# %option
# % key: granularity
# % type: string
# % description: Aggregation granularity, format absolute time "x years, x months, x weeks, x days, x hours, x minutes, x seconds" or an integer value for relative time
# % required: yes
# % multiple: no
# %end

# %option G_OPT_T_SAMPLE
# % options: equal,overlaps,overlapped,starts,started,finishes,finished,during,contains
# % answer: contains
# %end

# %option G_OPT_M_NPROCS
# %end

# %option
# % key: region_relation
# % description: Process only maps with this spatial relation to the current computational region
# % guisection: Selection
# % options: overlaps,contains,is_contained
# % required: no
# % multiple: no
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# % key: n
# % description: Register Null maps
# %end

# %flag
# % key: z
# % description: Use zero (0) for transparency instead of NULL
# %end

# %flag
# % key: s
# % description: Do not create color and category files
# %end

# %flag
# % key: v
# % description: Patch to virtual raster map (r.buildvrt)
# %end

# %rules
# % excludes: -v,-s,-z
# % collective: title,description
# % required: -e,title,description
# %end

import sys
from copy import deepcopy

import grass.script as gs
import grass.pygrass.modules as pymod
import grass.temporal as tgis

from grass.temporal.space_time_datasets import RasterDataset
from grass.temporal.datetime_math import (
    create_suffix_from_datetime,
    create_time_suffix,
    create_numeric_suffix,
)
from grass.temporal.core import (
    get_current_mapset,
    get_tgis_message_interface,
    init_dbif,
)
from grass.temporal.open_stds import open_old_stds
from grass.temporal.spatio_temporal_relationships import SpatioTemporalTopologyBuilder


def patch_by_topology(
    granularity_list,
    granularity,
    map_list,
    topo_list,
    basename,
    time_suffix,
    offset=0,
    module="r.patch",
    nprocs=1,
    sort="asc",
    dbif=None,
    patch_flags="",
    overwrite=False,
):
    """Aggregate a list of raster input maps with r.series

    :param granularity_list: A list of AbstractMapDataset objects.
                             The temporal extents of the objects are used
                             to build the spatio-temporal topology with the
                             map list objects
    :param granularity: The granularity of the granularity list
    :param map_list: A list of RasterDataset objects that contain the raster
                     maps that should be aggregated
    :param topo_list: A list of strings of topological relations that are
                      used to select the raster maps for aggregation
    :param basename: The basename of the new generated raster maps
    :param time_suffix: Use the granularity truncated start time of the
                        actual granule to create the suffix for the basename
    :param offset: Use a numerical offset for suffix generation
                   (overwritten by time_suffix)
    :param module: The GRASS GIS module to use for aggregation (r.patch or r.buildvrt)
    :param nprocs: The number of processes used for parallel computation (only used with )
    :param sort: Sort order for raster maps send to r.patch/r.buildvrt
    :param dbif: The database interface to be used
    :param patch_flags: Flags set for patch module ("", "z" or "s")
    :param overwrite: Overwrite existing raster maps
    :return: A list of RasterDataset objects that contain the new map names
             and the temporal extent as well as semantic_labels for map registration
    """
    msgr = get_tgis_message_interface()

    dbif, connection_state_changed = init_dbif(dbif)

    if not map_list:
        return None

    map_dict = {}

    for row in map_list:
        # semantic_label = row.semantic_label
        if row.metadata.semantic_label in map_dict:
            map_dict[row.metadata.semantic_label].append(row)
        else:
            map_dict[row.metadata.semantic_label] = [row]

    # The module queue for parallel execution
    process_queue = pymod.ParallelModuleQueue(nprocs)

    # Dummy process object that will be deep copied
    # and be put into the process queue
    patch_module = pymod.Module(
        module,
        overwrite=overwrite,
        input="ordered_raster_list",
        output="output",
        flags=patch_flags,
        quiet=True,
        run_=False,
        finish_=False,
    )

    # Copy map if only one map is found in granule
    copy_module = pymod.Module(
        "g.copy", raster=["spam", "spamspam"], quiet=True, run_=False, finish_=False
    )

    output_list = []
    count = 0
    current_mapset = get_current_mapset()

    for semantic_label, map_list in map_dict.items():

        topo_builder = SpatioTemporalTopologyBuilder()
        topo_builder.build(mapsA=granularity_list, mapsB=map_list)

        for granule in granularity_list:
            msgr.percent(count, len(granularity_list), 1)
            count += 1

            aggregation_list = []

            # Handle semantic labels (one granule per semantic label)
            if "equal" in topo_list and granule.equal:
                for map_layer in granule.equal:
                    aggregation_list.append(map_layer.get_name())
            if "contains" in topo_list and granule.contains:
                for map_layer in granule.contains:
                    aggregation_list.append(map_layer.get_name())
            if "during" in topo_list and granule.during:
                for map_layer in granule.during:
                    aggregation_list.append(map_layer.get_name())
            if "starts" in topo_list and granule.starts:
                for map_layer in granule.starts:
                    aggregation_list.append(map_layer.get_name())
            if "started" in topo_list and granule.started:
                for map_layer in granule.started:
                    aggregation_list.append(map_layer.get_name())
            if "finishes" in topo_list and granule.finishes:
                for map_layer in granule.finishes:
                    aggregation_list.append(map_layer.get_name())
            if "finished" in topo_list and granule.finished:
                for map_layer in granule.finished:
                    aggregation_list.append(map_layer.get_name())
            if "overlaps" in topo_list and granule.overlaps:
                for map_layer in granule.overlaps:
                    aggregation_list.append(map_layer.get_name())
            if "overlapped" in topo_list and granule.overlapped:
                for map_layer in granule.overlapped:
                    aggregation_list.append(map_layer.get_name())

            if aggregation_list:
                msgr.verbose(
                    _(
                        "Aggregating {n} raster maps from '{start}' to '{end}'"
                        " with semantic label '{semantic_label}'"
                    ).format(
                        n=len(aggregation_list),
                        start=str(granule.temporal_extent.get_start_time()),
                        end=str(granule.temporal_extent.get_end_time()),
                        semantic_label=semantic_label,
                    )
                )

                if granule.is_time_absolute() is True and time_suffix == "gran":
                    suffix = create_suffix_from_datetime(
                        granule.temporal_extent.get_start_time(), granularity
                    )
                elif granule.is_time_absolute() is True and time_suffix == "time":
                    suffix = create_time_suffix(granule)

                else:
                    suffix = create_numeric_suffix(
                        "", count + int(offset), time_suffix
                    ).removeprefix("_")
                output_name = (
                    f"{basename}_{semantic_label}_{suffix}"
                    if semantic_label
                    else f"{basename}_{suffix}"
                )

                map_layer = RasterDataset(f"{output_name}@{current_mapset}")
                map_layer.set_temporal_extent(granule.get_temporal_extent())
                map_layer.set_semantic_label(semantic_label)

                if map_layer.map_exists() is True and overwrite is False:
                    msgr.fatal(
                        _(
                            "Unable to perform aggregation. Output raster "
                            "map <{name}> exists and overwrite flag was "
                            "not set"
                        ).format(name=output_name)
                    )

                output_list.append(map_layer)

                if sort == "desc":
                    aggregation_list.reverse()
                if len(aggregation_list) > 1:
                    # Create the r.patch / r.buildvrt module
                    mod = deepcopy(patch_module)
                    mod(input=",".join(aggregation_list[::-1]), output=output_name)
                else:
                    # Create the g.copy module for single input maps
                    mod = deepcopy(copy_module)
                    mod(raster=[aggregation_list[0], output_name])

                process_queue.put(mod)

    process_queue.wait()

    if connection_state_changed:
        dbif.close()

    msgr.percent(1, 1, 1)

    return output_list


def main():
    """Main function"""
    # lazy imports
    overwrite = gs.overwrite()

    # Get the options
    gran = options["granularity"]

    # Make sure the temporal database exists
    tgis.init()

    dbif = tgis.SQLDatabaseInterfaceConnection()
    dbif.connect()
    current_mapset = get_current_mapset()

    spatial_extent = None
    if options["region_relation"]:
        spatial_extent = gs.parse_command("g.region", flags="3gu")
    sp = open_old_stds(options["input"], "strds")

    # We will create the strds later, but need to check here
    tgis.check_new_stds(options["output"], "strds", dbif, overwrite)

    map_list = sp.get_registered_maps_as_objects(
        where=options["where"],
        order="start_time",
        dbif=dbif,
        spatial_extent=spatial_extent,
        spatial_relation=options["region_relation"],
    )

    if not map_list:
        gs.warning(
            _("No maps found to process in Space time raster dataset <{}>.").format(
                options["input"]
            )
        )
        dbif.close()
        sys.exit(0)

    patch_flags = ""
    if flags["z"]:
        patch_flags += "z"
    if flags["s"]:
        patch_flags += "s"

    start_time = map_list[0].temporal_extent.get_start_time()

    if sp.is_time_absolute():
        start_time = tgis.adjust_datetime_to_granularity(start_time, gran)

    # We use the end time first
    end_time = map_list[-1].temporal_extent.get_end_time()
    has_end_time = True

    # In case no end time is available, then we use the start time of the last map layer
    if end_time is None:
        end_time = map_list[-1].temporal_extent.get_start_time()
        has_end_time = False

    granularity_list = []

    # Build the granularity list
    while True:
        if has_end_time is True:
            if start_time >= end_time:
                break
        else:
            if start_time > end_time:
                break

        granule = tgis.RasterDataset(None)
        start = start_time
        if sp.is_time_absolute():
            end = tgis.increment_datetime_by_string(start_time, gran)
            granule.set_absolute_time(start, end)
        else:
            end = start_time + int(gran)
            granule.set_relative_time(start, end, sp.get_relative_time_unit())
        start_time = end

        granularity_list.append(granule)

    output_list = patch_by_topology(
        granularity_list=granularity_list,
        granularity=gran,
        map_list=map_list,
        topo_list=options["sampling"].split(","),
        basename=options["basename"],
        time_suffix=options["suffix"],
        offset=options["offset"],
        module="r.buildvrt" if flags["v"] else "r.patch",
        nprocs=int(options["nprocs"]),
        sort=options["sort"],
        overwrite=overwrite,
    )
    if output_list:
        temporal_type, semantic_type, title, description = sp.get_initial_values()
        # Initialize SpaceTimeRasterDataset (STRDS) using tgis
        strds_long_name = f"{options['output']}@{current_mapset}"
        output_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

        # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
        if output_strds.is_in_db() and not overwrite:
            gs.fatal(
                _(
                    "Output STRDS <{}> exists."
                    "Use --overwrite together with -e to modify the existing STRDS."
                ).format(options["output"])
            )

        # Create STRDS if needed
        if not output_strds.is_in_db() or (overwrite and not flags["e"]):
            output_strds = tgis.open_new_stds(
                options["output"],
                "strds",
                temporal_type,
                options["title"],
                options["description"],
                semantic_type,
                dbif,
                overwrite,
            )
        else:
            output_strds = open_old_stds(options["input"], "strds")

        tgis.register_map_object_list(
            "rast",
            output_list,
            output_strds,
            flags["n"] is not True,
            sp.get_relative_time_unit(),
            dbif,
        )

        # Update the raster metadata table entries with aggregation type
        # output_strds.set_aggregation_type(method)
        output_strds.metadata.update(dbif)

    dbif.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    main()
