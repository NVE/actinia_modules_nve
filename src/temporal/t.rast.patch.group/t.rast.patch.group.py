#!/usr/bin/env python3

"""
# MODULE:    t.rast.patch
# AUTHOR(S):    Anika Bettge, Stefan Blumentrath
#
# PURPOSE:    Patch rasters maps in a space time raster dataset grouped by
              semantic label and temporal granule
# COPYRIGHT:    (C) 2019 by by mundialis and the GRASS Development Team
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
"""

# %module
# % description: Patch rasters maps in a space time raster dataset grouped by semantic label and temporal granule
# % keyword: temporal
# % keyword: aggregation
# % keyword: series
# % keyword: raster
# % keyword: merge
# % keyword: patch
# % keyword: time
# % keyword: granule
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_STRDS_OUTPUT
# %end

# %option
# % key: granularity
# % type: string
# % description: Aggregation granularity, format absolute time "x years, x months, x weeks, x days, x hours, x minutes, x seconds" or an integer value for relative time
# % required: no
# % multiple: no
# %end

# %option
# % key: semantic_labels
# % type: string
# % description: Comma separated list of Semantic labels to process (default is all)
# % required: no
# % multiple: yes
# %end

# %option
# % key: basename
# % type: string
# % label: Basename of the new generated output maps (default is common prefix of input maps)
# % description: Either a numerical suffix or the start time (s-flag) separated by an underscore will be attached to create a unique identifier
# % required: no
# % multiple: no
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

# %option
# % key: sort
# % description: Sort order (see sort parameter)
# % options: asc,desc
# % answer: desc
# %end

# %rules
# % excludes: -v,-s,-z
# %end

import grass.script as gs
from grass.exceptions import CalledModuleError


def patch_maps(ordered_rasts, patch_flags, output_map, patch_module="r.patch"):
    Module(
        patch_module,
        overwrite=gs.overwrite(),
        input=(",").join(ordered_rasts),
        output=output_map,
        flags=patch_flags,
    )


def main():
    # lazy imports
    import gs.temporal as tgis

    # Get the options
    input = options["input"]
    output = options["output"]
    where = options["where"]
    sort = options["sort"]
    add_time = flags["t"]
    patch_s = flags["s"]
    patch_z = flags["z"]
    patch_module = "r.buildvrt" if flags["v"] else "r.patch"

    # Make sure the temporal database exists
    dbif = tgis.init()

    rows = sp.get_registered_maps("id", where, "start_time", None)
    # Get list of maps in input STRDS
    input_strds = tgis.open_old_stds(options["input"], "strds", dbif)

    # Check semantic labels

    where = options["where"]

    map_rows = input_strds.get_registered_maps(
        "id,start_time,end_time,semantic_label",
        where,
        "start_time",
        dbif,
    )

    # Group maps by granularity and / or semantic label
    gran = options["granularity"]

    map_groups = {}
    for row in map_rows:
        if gran:
            start_time = tgis.adjust_datetime_to_granularity(row["start_time"], gran)
            end_time = tgis.increment_datetime_by_string(start_time, gran)
        else:
            start_time = row["start_time"]
            end_time = row["end_time"] or row["start_time"]
        if row["semantic_label"] not in map_groups:
            map_groups[row["semantic_label"]] = [
                {"start_time": start_time, "end_time": end_time, "maps": row["id"]}
            ]
        else:
            map_groups[row["semantic_label"]].append(
                {"start_time": start_time, "end_time": end_time, "maps": [row["id"]]}
            )
    if not gran:
        map_groups[row["semantic_label"]] = {
            "start_time": start_time,
            "end_time": end_time,
            "maps": [row["id"]],
        }

        if (start_time, end_time) not in map_groups:
            map_groups[(start_time, end_time)] = {row["semantic_label"]: [row["id"]]}
        else:
            map_groups[row["semantic_label"]] = {}
        if gran:
            # else:
            if row["semantic_label"] in map_groups[(start_time, end_time)]:
                map_groups[(start_time, end_time)][row["semantic_label"]].append(
                    row["id"]
                )
            else:
                map_groups[(start_time, end_time)][row["semantic_label"]] = [row["id"]]

        elif (row["start_time"], row["end_time"]) not in map_groups:
            map_groups[(row["start_time"], row["end_time"])] = {
                row["semantic_label"]: row["id"]
            }
        else:
            map_groups[(row["start_time"], row["end_time"])][row["semantic_label"]] = (
                row["id"]
            )

    if not rows:
        gs.warning(_("No maps found to process"))
        sys.exit(0)
    ordered_rasts = []
    # newest images are first
    if sort == "desc":
        rows_sorted = rows[::-1]
    # older images are first
    elif sort == "asc":
        rows_sorted = rows

    for row in rows_sorted:
        string = str(row["id"])
        ordered_rasts.append(string)

    patch_flags = ""
    if patch_z:
        patch_flags += "z"
    if patch_s:
        patch_flags += "s"

    try:
        gs.run_command(
            patch_module,
            overwrite=gs.overwrite(),
            input=(",").join(ordered_rasts),
            output=output,
            flags=patch_flags,
        )
    except CalledModuleError:
        gs.fatal(_("{} failed. Check above error messages.").format(patch_module))

    if not add_time:
        # We need to set the temporal extent from the subset of selected maps
        maps = sp.get_registered_maps_as_objects(
            where=where, order="start_time", dbif=None
        )
        first_map = maps[0]
        last_map = maps[-1]
        start_a, end_a = first_map.get_temporal_extent_as_tuple()
        start_b, end_b = last_map.get_temporal_extent_as_tuple()

        if end_b is None:
            end_b = start_b

        if first_map.is_time_absolute():
            extent = tgis.AbsoluteTemporalExtent(start_time=start_a, end_time=end_b)
        else:
            extent = tgis.RelativeTemporalExtent(
                start_time=start_a,
                end_time=end_b,
                unit=first_map.get_relative_time_unit(),
            )

        # Create the time range for the output map
        if output.find("@") >= 0:
            id = output
        else:
            mapset = gs.gisenv()["MAPSET"]
            id = output + "@" + mapset

        map = sp.get_new_map_instance(id)
        map.load()

        map.set_temporal_extent(extent=extent)

        # Register the map in the temporal database
        if map.is_in_db():
            map.update_all()
        else:
            map.insert()


if __name__ == "__main__":
    options, flags = gs.parser()
    main()
