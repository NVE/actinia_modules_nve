#!/usr/bin/env python

"""
MODULE:    t.rast.aggregate.condition
AUTHOR(S): Stefan Blumentrath

PURPOSE:   Aggregates rasters maps in space and time, applying a condition for valid data using r.mapcalc
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
# % description: Aggregate multiple space time raster maps into mosaics with the given granualrity, applying a condition for valid data using r.mapcalc.
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
# % guisection: Selection
# %end

# %option
# % key: mask_label
# % multiple: no
# % description: Semantic label of the mask raster maps
# % required: no
# %end

# %option
# % key: mask_value
# % multiple: no
# % description: Value of the mask raster maps representing valid data
# % required: yes
# %end

# %option
# % key: condition_label
# % multiple: no
# % description: Semantic label of the condition raster maps
# % required: yes
# %end

# %option
# % key: aggregation_labels
# % multiple: yes
# % description: One ore more semantic label(s) of raster map(s) to aggregate
# % required: yes
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
# % required: yes
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
# % guisection: Selection
# %end

# %option
# % key: aggregate_condition
# % description: Condition used to identify/select values of maps to aggregate (NULL aware method from r.mapcalc)
# % options: nmax,nmin,nmode
# % required: yes
# % multiple: no
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


# %rules
# % collective: title,description
# % required: -e,title,description
# %end

# ToDo:
# - Support granules from moving windows (even if leading to invalid temporal topology) with:
#   m-flag for moving temporal window + d-flag for decrementing granule (default is increment)

# - implement n-flag
# - Create a TemporalExtentTuple class based on
#   https://grass.osgeo.org/grass84/manuals/libpython/_modules/grass/temporal/temporal_extent.html
#   That would improve the performance if no more advanced temporal objects are needed
# - add more statistics output (e.g. number of valid pixels aggregated, max/min in valid pixels)
# - make mask optional (do not require masking)
# - allow empty (or *) mask_value, if * all values in mask raster are considered valid data

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


def create_ganule_list(map_list, granularity, relative_time_unit=None):
    """Create a list of empty RasterDataset with the given temporal
    granularity from a list of input maps
    :param map_list: List of database rows (SQLite or PostgreSQL)
    :param granularity: string describing the granularity of the output list,
                        is expected to be validated beforehand
    :relative_time_unit: string with the relative time unit of the input
                         STRDS, None means absolute time
    :return granularity_list: a list of RasterDataset with temporal extent"""
    start_time = map_list[0]["start_time"]

    if not relative_time_unit:
        start_time = tgis.adjust_datetime_to_granularity(start_time, granularity)

    # We use the end time first
    end_time = map_list[-1]["end_time"]
    has_end_time = True

    # In case no end time is available, then we use the start time of the last map layer
    if end_time is None:
        end_time = map_list[-1]["start_time"]
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
        if relative_time_unit:
            # For input STRDS with relative time
            end = start_time + int(granularity)
            granule.set_relative_time(start, end, relative_time_unit)
        else:
            # For input STRDS with absolute time
            end = tgis.increment_datetime_by_string(start_time, granularity)
            granule.set_absolute_time(start, end)
        start_time = end

        granularity_list.append(granule)
    return granularity_list


def aggregate_with_condition(
    granularity_list,
    granularity,
    map_list,
    time_unit=None,
    basename=None,
    time_suffix="gran",
    offset=0,
    topo_list=None,
    mask_label=None,
    mask_value=0,
    condition_label=None,
    aggregate_condition="nmax",
    aggregation_labels=None,
    nprocs=1,
    dbif=None,
):
    """Aggregate a list of raster input maps with r.mapcalc

    :param granularity_list: A list of AbstractMapDataset objects.
                             The temporal extents of the objects are used
                             to build the spatio-temporal topology with the
                             map list objects
    :param granularity: The granularity of the granularity list
    :param map_list: A list of RasterDataset objects that contain the raster
                     maps that should be aggregated
    :param time_unit: Relative time unit of the maps in map_list
    :param basename: The basename of the new generated raster maps
    :param time_suffix: Use the granularity truncated start time of the
                        actual granule to create the suffix for the basename
    :param offset: Use a numerical offset for suffix generation
                   (overwritten by time_suffix)
    :param topo_list: A list of strings of topological relations that are
                      used to select the raster maps for aggregation
    :param mask_label: Semantic label that represent mask maps
    :param mask_value: integer value representing valid data in the mask raster maps
    :param condition_label: Semantic label that represent maps that define the condition
                            for selecting putput pixels
    :param aggregate_condition: string of the r.mapcalc method used for
                                aggregating the condition maps (default is nmax)
    :param aggregation_labels: Semantic labels that represent maps to be aggregated
    :param nprocs: The number of processes used for parallel computation (only used with )
    :param dbif: The database interface to be used
    :return: A list of RasterDataset objects that contain the new map names
             and the temporal extent as well as semantic_labels for map registration
    """

    if not map_list:
        return None

    if not topo_list:
        topo_list = ["contains"]

    msgr = get_tgis_message_interface()

    dbif, connection_state_changed = init_dbif(dbif)

    agg_module = pymod.Module(
        "r.mapcalc",
        overwrite=gs.overwrite(),
        quiet=True,
        run_=False,
        # finish_=False,
    )

    count = 0
    output_list = []
    current_mapset = get_current_mapset()

    # The module queue for parallel execution
    process_queue = pymod.ParallelModuleQueue(nprocs)

    map_dict = {}
    for raster_maps in map_list:
        raster_map = tgis.RasterDataset(None)
        if time_unit:
            raster_map.set_relative_time(
                raster_maps["start_time"], raster_maps["end_time"], time_unit
            )
        else:
            raster_map.set_absolute_time(
                raster_maps["start_time"], raster_maps["end_time"]
            )

        map_dict[raster_map] = {
            "id": raster_maps["ids"],
            "semantic_label": raster_maps["semantic_labels"],
        }

    topo_builder = SpatioTemporalTopologyBuilder()
    topo_builder.build(mapsA=granularity_list, mapsB=list(map_dict.keys()))

    res_dict_template = {
        "condition_labels": [],  # Condition label
        "mask_labels": [],  # Mask label
    }
    for granule in granularity_list:
        msgr.percent(count, len(granularity_list), 1)
        count += 1

        granule_temporal_extent = granule.get_temporal_extent()

        for aggregation_label in aggregation_labels:
            res_dict_template[aggregation_label] = []

        res_dict = deepcopy(res_dict_template)

        # Loop over maps with matching temporal topology
        for topology in topo_list:
            matching_objects = getattr(granule, topology)
            # Check if any maps are temporally related to the granule with the given topology
            if matching_objects:
                for matching_object in matching_objects:
                    map_ids = map_dict[matching_object]["id"].split(",")
                    semantic_labels = map_dict[matching_object]["semantic_label"].split(
                        ","
                    )
                    if len(map_ids) != len(semantic_labels):
                        gs.warning("Missing maps")
                        continue
                    if not set(
                        [mask_label, condition_label, *aggregation_labels]
                    ).issubset(set(semantic_labels)):
                        gs.warning(
                            _(
                                "Missing input some raster maps for {extent}. Found only the following semantic_labels: {labels}"
                            ).format(
                                extent=" - ".join(
                                    [
                                        dt.isoformat()
                                        for dt in matching_object.get_absolute_time()
                                    ]
                                ),
                                labels=(
                                    ",".join(semantic_labels)
                                    if semantic_labels
                                    else None
                                ),
                            )
                        )
                        continue

                    # Create mask expression for condition map
                    mask_map = map_ids[semantic_labels.index(mask_label)]
                    condition_map = map_ids[semantic_labels.index(condition_label)]
                    mask_list = f"if({mask_map}=={mask_value},{condition_map},null())"

                    # Create mask expression for aggregation map
                    for aggregation_label in aggregation_labels:
                        res_dict[aggregation_label].append(
                            f"if({mask_map}=={mask_value},if({{output_condition_map}}=={condition_map},{map_ids[semantic_labels.index(aggregation_label)]},null()),null())"
                        )
                    res_dict["mask_labels"].append(mask_list)

        # Check if any maps are temporally related to the granule
        if res_dict != res_dict_template:
            if granule.is_time_absolute() is True and time_suffix == "gran":
                suffix = create_suffix_from_datetime(
                    granule.temporal_extent.get_start_time(), granularity
                )
            elif granule.is_time_absolute() is True and time_suffix == "time":
                suffix = create_time_suffix(granule)

            else:
                suffix = create_numeric_suffix(
                    "", count + offset, time_suffix
                ).removeprefix("_")
            output_name = f"{basename}_{suffix}"

            # Compile expressions
            expression = f"{output_name}_{condition_label}_{aggregate_condition}={aggregate_condition}({','.join(res_dict['mask_labels'])})\n"
            map_layer = RasterDataset(
                f"{output_name}_{condition_label}_{aggregate_condition}@{current_mapset}"
            )
            map_layer.set_temporal_extent(granule_temporal_extent)
            map_layer.set_semantic_label(f"{condition_label}_{aggregate_condition}")
            output_list.append(map_layer)
            condition_module = deepcopy(agg_module)
            condition_module.inputs.expression = expression
            expression = ""
            for aggregation_label in aggregation_labels:
                expression += f"{output_name}_{aggregation_label}=nmedian({','.join([eval_expression for eval_expression in res_dict[aggregation_label]])})"
                map_layer = RasterDataset(
                    f"{output_name}_{aggregation_label}@{current_mapset}"
                )
                map_layer.set_temporal_extent(granule_temporal_extent)
                map_layer.set_semantic_label(aggregation_label)
                output_list.append(map_layer)
            expression = expression.format(
                output_condition_map=f"{output_name}_{condition_label}_{aggregate_condition}"
            )

            mc_module = deepcopy(agg_module)
            mc_module.inputs.expression = expression.format(
                output_condition_map=f"{output_name}_{condition_label}_{aggregate_condition}"
            )

            # Add modules to process queue
            process_queue.put(pymod.MultiModule([condition_module, mc_module]))
    process_queue.wait()

    if connection_state_changed:
        dbif.close()

    msgr.percent(1, 1, 1)

    return output_list


def get_registered_maps_grouped(
    stds,
    columns=None,
    where=None,
    group_by=None,
    spatial_extent=None,
    spatial_relation=None,
    dbif=None,
):
    """Return SQL rows of all registered maps.

    In case columns are not specified, each row includes all columns
    specified in the datatype specific view.

    The combination of the spatial_extent and spatial_relation parameters
    can be used to return only SQL rows of maps with the given spatial
    relation to the provided spatial extent

    :param columns: Columns to be selected as list of SQL compliant strings
    :param where: The SQL where statement to select a subset
                    of the registered maps without "WHERE"
    :param group_by: The columns to be used in the SQL GROUP BY statement
                      as list of SQL compliant strings
    :param dbif: The database interface to be used
    :param spatial_extent: Spatial extent dict and projection information
        e.g. from g.region -ug3 with GRASS GIS region keys
        "n", "s", "e", "w", "b", "t", and  "projection".
    :param spatial_relation: Spatial relation to the provided
        spatial extent as a string with one of the following values:
        "overlaps": maps that spatially overlap ("intersect")
                    within the provided spatial extent
        "is_contained": maps that are fully within the provided spatial extent
        "contains": maps that contain (fully cover) the provided spatial extent

    :return: SQL rows of all registered maps,
            In case nothing found None is returned
    """

    dbif, connection_state_changed = init_dbif(dbif)

    if not columns:
        columns = ["id", "semantic_label"]

    if not group_by:
        group_by = ["start_time", "end_time"]

    rows = None

    if stds.get_map_register() is not None:
        # Use the correct temporal table
        if stds.get_temporal_type() == "absolute":
            map_view = stds.get_new_map_instance(None).get_type() + "_view_abs_time"
        else:
            map_view = stds.get_new_map_instance(None).get_type() + "_view_rel_time"

        group_columns = ",".join(group_by)
        columns = (
            group_columns
            + ", "
            + ", ".join(
                [f"group_concat({column},',') AS {column}s" for column in columns]
            )
        )

        # filter by spatial extent
        if spatial_extent and spatial_relation:
            where = stds._update_where_statement_by_spatial_extent(
                where, spatial_extent, spatial_relation
            )

        sql = "SELECT %s FROM %s  WHERE %s.id IN (SELECT id FROM %s)" % (
            columns,
            map_view,
            map_view,
            stds.get_map_register(),
        )

        if where is not None and where != "":
            sql += " AND (%s)" % (where.split(";")[0])
        sql += f" GROUP BY {group_columns};"
        try:
            dbif.execute(sql, mapset=stds.base.mapset)
            rows = dbif.fetchall(mapset=stds.base.mapset)
        except:
            if connection_state_changed:
                dbif.close()
            stds.msgr.error(
                _("Unable to get map ids from register table <{}>").format(
                    stds.get_map_register()
                )
            )
            raise

    if connection_state_changed:
        dbif.close()

    return rows


def main():
    """Main function"""
    # lazy imports
    overwrite = gs.overwrite()

    # Get the options
    where = options["where"]

    # Make sure the temporal database exists
    tgis.init()

    dbif = tgis.SQLDatabaseInterfaceConnection()
    dbif.connect()
    current_mapset = get_current_mapset()

    spatial_extent = None
    if options["region_relation"]:
        spatial_extent = gs.parse_command("g.region", flags="3gu")
    input_strds = open_old_stds(options["input"], "strds")

    # We will create the strds later, but need to check here
    tgis.check_new_stds(options["output"], "strds", dbif, overwrite)
    relative_time_unit = input_strds.get_relative_time_unit()

    # Get and check semantic labels
    semantic_labels = [
        options["condition_label"],
        options["mask_label"],
        *options["aggregation_labels"].split(","),
    ]
    missing_labels = set(semantic_labels).difference(
        input_strds.metadata.semantic_labels.split(",")
    )
    if missing_labels:
        gs.fatal(
            _("Semantic labels <{labels}> are missing from STRDS <{strds}>").format(
                strds=input_strds.get_id(), labels=", ".join(missing_labels)
            )
        )
    semantic_labels = ",".join(
        [f"'{semantic_label}'" for semantic_label in semantic_labels]
    )

    if where:
        where = f"{where} AND semantic_label in ({semantic_labels})"
    else:
        where = f"semantic_label in ({semantic_labels})"

    # Get a list of input maps
    map_list = get_registered_maps_grouped(
        input_strds,
        where=where,
        spatial_relation=options["region_relation"],
        spatial_extent=spatial_extent,
        dbif=dbif,
    )

    if not map_list:
        gs.warning(
            _("No maps found to process in Space time raster dataset <{}>.").format(
                options["input"]
            )
        )
        dbif.close()
        sys.exit(0)

    # Create granule list from map list
    granularity_list = create_ganule_list(
        map_list, options["granularity"], relative_time_unit=relative_time_unit
    )

    output_list = aggregate_with_condition(
        granularity_list,
        options["granularity"],
        map_list,
        time_unit=relative_time_unit,
        basename=options["basename"],
        offset=options["offset"],
        nprocs=int(options["nprocs"]),
        topo_list=options["sampling"].split(","),
        aggregate_condition=options["aggregate_condition"],
        time_suffix=options["suffix"],
        mask_value=options["mask_value"],
        condition_label=options["condition_label"],
        mask_label=options["mask_label"],
        aggregation_labels=options["aggregation_labels"].split(","),
        dbif=dbif,
    )

    if output_list:
        (
            temporal_type,
            semantic_type,
            title,
            description,
        ) = input_strds.get_initial_values()
        title = options["title"] or title
        description = options["description"] or description

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
                title,
                description,
                semantic_type,
                dbif,
                overwrite,
            )
        else:
            output_strds = open_old_stds(options["output"], "strds")

        tgis.register_map_object_list(
            "rast",
            output_list,
            output_strds,
            flags["n"] is not True,
            relative_time_unit,
            dbif,
        )

        # Update the raster metadata table entries with aggregation type
        # output_strds.set_aggregation_type(method)
        output_strds.metadata.update(dbif)

    dbif.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    main()
