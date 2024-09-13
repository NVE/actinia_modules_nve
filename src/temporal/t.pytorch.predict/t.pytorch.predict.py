#!/usr/bin/env python3

"""
MODULE:      t.pytorch.predict
AUTHOR(S):   Stefan Blumentrath
PURPOSE:     Apply a pytorch model to imagery groups in a Space Time Raster Dataset
             and register results in an output STRDS
COPYRIGHT:   (C) 2023-2024 by Norwegian Water and Energy Directorate
             (NVE), Stefan Blumentrath and the GRASS GIS Development Team

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

ToDo:
- tiling from vector map (to avoid unnecessary data reads outside core AOI)
- test case
"""

# %Module
# % description: Apply a pytorch model to imagery groups in a Space Time Raster Dataset (STRDS)
# % keyword: temporal
# % keyword: machine learning
# % keyword: deep learning
# % keyword: pytorch
# % keyword: unet
# % keyword: GPU
# % keyword: predict
# % keyword: imagery
# % keyword: raster
# % keyword: strds
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_STRDS_INPUTS
# %key: reference_strds
# % required: no
# %end

# %option G_OPT_T_WHERE
# %key: reference_where
# % required: no
# % description: Where clause to select reference images
# %end

# %option
# %key: reference_suffix
# % description: Suffix to be added to the semantic label of the raster maps in the reference_strds
# # % type: string
# # % required: no
# # % multiple: no
# %end

# %option G_OPT_T_SAMPLE
# # % required: no
# # % multiple: no
# %end

# %option
# % key: offset
# % type: integer
# % required: no
# % multiple: no
# % description: Offset that defines a reference map (e.g. -1 for the previous map (group) in the input STRDS)
# %end

# %option G_OPT_I_GROUP
# %key: auxillary_group
# % required: no
# % multiple: no
# % description: Input imagery group with time independent raster maps
# %end

# %option
# % key: region_relation
# % description: Process only maps with this spatial relation to the current computational region
# % options: overlaps,contains,is_contained
# % required: no
# % multiple: no
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

# %option G_OPT_V_INPUT
# % key: vector_tiles
# % required: no
# % description: Vector map with tiles to process (will be extended by "overlap")
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
# % key: c
# % description: Use CPU as device for prediction, default is use cuda (GPU) if detected
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# % key: l
# % description: Limit output to valid range (data outside the valid range is set to valid min/max)
# %end

# %flag
# % key: s
# % description: Skip incomplete groups (do not fail)
# %end

# # %flag
# # %key: m
# # % description: Match semantic labels between input and reference_strds
# # %end

# %rules
# % exclusive: offset,reference_strds
# % exclusive: tile_size,vector_tiles
# % collective: title,description
# % required: -e,title,description
# %end

import json
import os
import sys
from functools import partial
from itertools import starmap
from math import floor
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs
import grass.temporal as tgis
from grass.exceptions import CalledModuleError
from grass.pygrass.modules.interface import Module
from grass.temporal.register import register_maps_in_space_time_dataset

TMP_NAME = gs.tempname(12)
# Get GRASS GIS environment
GISENV = dict(gs.gisenv())


def group_to_dict(
    imagery_group_name,
    subgroup=None,
    dict_keys="semantic_labels",
    dict_values="map_names",
    fill_semantic_label=True,
    env=None,
):
    """Create a dictionary to represent an imagery group with metadata.

    Defined by the dict_keys option, the dictionary uses either the names
    of the raster maps ("map_names"), their row indices in the group
    ("indices") or their associated semantic_labels ("semantic_labels") as keys.
    The default is to use semantic_labels. Note that map metadata
    of the maps in the group have to be read to get the semantic label,
    in addition to the group file. The same metadata is read when the
    "metadata" is requested as dict_values. Other supported dict_values
    are "map_names" (default), "semantic_labels", or "indices".

    The function can also operate on the level of subgroups. In case a
    non-existing (or empty sub-group) is requested a warning is printed
    and an empty dictionary is returned (following the behavior of i.group).

    Example::

    >>> run_command("g.copy", raster="lsat7_2000_10,lsat7_2000_10")
    >>> run_command("r.support", raster="lsat7_2000_10", semantic_label="L8_1")
    >>> run_command("g.copy", raster="lsat7_2000_20,lsat7_2000_20")
    >>> run_command("r.support", raster="lsat7_2000_20", semantic_label="L8_2")
    >>> run_command("g.copy", raster="lsat7_2000_30,lsat7_2000_30")
    >>> run_command("r.support", raster="lsat7_2000_30", semantic_label="L8_3")
    >>> run_command("i.group", group="L8_group",
    >>>             input="lsat7_2000_10,lsat7_2000_20,lsat7_2000_30")
    >>> group_to_dict("L8_group")  # doctest: +ELLIPSIS
    {"L8_1": "lsat7_2000_10", ... "L8_3": "lsat7_2000_30"}
    >>> run_command("g.remove", flags="f", type="group", name="L8_group")
    >>> run_command("g.remove", flags="f", type="raster",
    >>>             name="lsat7_2000_10,lsat7_2000_20,lsat7_2000_30")

    :param str imagery_group_name: Name of the imagery group to process (or None)
    :param str subgroup: Name of the imagery sub-group to process (or None)
    :param str dict_keys: What to use as key for dictionary. Can bei either
                         "semantic_labels" (default), "map_names" or "indices"
    :param str dict_values: What to use as values for dictionary. Can bei either
                           "map_names" (default), "semanic_labels", "indices" or
                           "metadata" (to return dictionaries with full map metadata)
    :param bool fill_semantic_label: If maps in a group do not have a semantic
                                     label, their index in the group is used
                                     instead (default). Otherwise None / "none"
                                     is used.
    :param dict env: Environment to use when parsing the imagery group

    :return: dictionary representing an imagery group with it's maps and their
             semantic labels, row indices in the group, or metadata
    :rtype: dict
    """
    group_dict = {}
    try:
        maps_in_group = (
            gs.read_command(
                "i.group",
                group=imagery_group_name,
                subgroup=subgroup,
                flags="g",
                quiet=True,
                env=env,
            )
            .strip()
            .split()
        )
    except CalledModuleError:
        gs.fatal(_("Could not parse imagery group <{}>").format(imagery_group_name))

    if dict_keys not in ["indices", "map_names", "semantic_labels"]:
        raise ValueError(f"Invalid dictionary keys <{dict_keys}> requested")

    if dict_values not in ["indices", "map_names", "semantic_labels", "metadata"]:
        raise ValueError(f"Invalid dictionary values <{dict_values}> requested")

    if subgroup and not maps_in_group:
        gs.warning(
            _("Empty result returned for subgroup <{sg}> in group <{g}>").format(
                sg=subgroup, g=imagery_group_name
            )
        )

    for idx, raster_map in enumerate(maps_in_group):
        raster_map_info = None
        # Get raster metadata if needed
        if (
            dict_values in ["semantic_labels", "metadata"]
            or dict_keys == "semantic_labels"
        ):
            raster_map_info = gs.raster_info(raster_map, env=env)

        # Get key for dictionary
        if dict_keys == "indices":
            key = str(idx + 1)
        elif dict_keys == "map_names":
            key = raster_map
        elif dict_keys == "semantic_labels":
            key = raster_map_info["semantic_label"]
            if not key or key == '"none"':
                gs.warning(
                    _(
                        "Raster map {m} in group <{g}> does not have a semantic label."
                    ).format(m=raster_map, g=imagery_group_name)
                )
                if fill_semantic_label:
                    key = str(idx + 1)

        if dict_values == "indices":
            val = str(idx + 1)
        elif dict_values == "map_names":
            val = raster_map
        elif dict_values == "semantic_labels":
            val = raster_map_info or raster_map
        elif dict_values == "metadata":
            val = raster_map_info
        if key in group_dict:
            gs.warning(
                _(
                    "Key {k} from raster map {m} already present in group dictionary."
                    "Overwriting existing entry..."
                ).format(k=key, r=raster_map)
            )
        group_dict[key] = val
    return group_dict


def is_int(string):
    """Check if a string represents an integer value"""
    try:
        int(string)
    except ValueError:
        return False
    return True


def get_registered_maps(
    stds,
    columns=None,
    where=None,
    group_by=None,
    spatial_extent=None,
    spatial_relation=None,
    dbif=None,
):
    """Return SQL rows of the selected registered maps, grouped by
    the columns in the group_by option.

    This function is useful to retrieve e.g. granules from an STRDS
    with satellite imagery where scene consists of different bands
    that have different semantic_labels but equal an temporal extend.

    The returned SQL rows contain the selected columns plus the columns
    in the group_by option are always included too. Content of the
    selected columns is concatenated to a comma separated string.

    The combination of the spatial_extent and spatial_relation parameters
    can be used to return only SQL rows of maps with the given spatial
    relation to the provided spatial extent

    :param columns: Columns to be selected as list of SQL compliant strings,
                    default is ["id", "semantic_label"].
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

    :return: SQL rows of all registered maps grouped by the columns given in
             the group_by option, in case no maps are found, None is returned
    """

    dbif, connection_state_changed = tgis.init_dbif(dbif)

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


def map_row_to_dict(map_row):
    """Create a dictionary from a group row returned by
    `get_registered_maps`

    :param map_row: SQL row returned by `get_registered_maps`
                    with keys: ids, semantic_labels
    """
    map_ids = map_row["ids"].split(",")
    if not map_row["semantic_labels"]:
        gs.warning("Semantic labels are missing for all raster maps")
        semantic_labels = [""] * len(map_ids)
    else:
        semantic_labels = map_row["semantic_labels"].split(",")
    return {semantic_labels[idx]: map_id for idx, map_id in enumerate(map_ids)}


def build_group_dict(
    map_list,
    time_unit=None,
    offset=None,
    raster_dataset=False,
):
    """Build dictionary to represent image groups registered in an STRDS"""
    maps_dict = {}
    for idx, raster_map_row in enumerate(map_list):
        group_dict = {}
        reference_dict = {}
        temporal_extent = [raster_map_row["start_time"], raster_map_row["end_time"]]
        if offset and 0 <= idx + offset < len(map_list):
            temporal_extent.extend(
                [
                    map_list[idx + offset]["start_time"],
                    map_list[idx + offset]["end_time"],
                ]
            )
            reference_dict = map_row_to_dict(map_list[idx + offset])
        # Remove potential None
        temporal_extent = {time_stamp for time_stamp in temporal_extent if time_stamp}
        temporal_extent = (
            min(temporal_extent),
            max(temporal_extent) if len(temporal_extent) > 1 else None,
        )
        if raster_dataset:
            # Use TGIS RasterDataset as dict key (for spatio-temporal relations)
            dict_key = tgis.RasterDataset(None)
            if time_unit:
                dict_key.set_relative_time(*temporal_extent, time_unit)
            else:
                dict_key.set_absolute_time(*temporal_extent)
        else:
            dict_key = temporal_extent

        group_dict = map_row_to_dict(raster_map_row)
        if reference_dict or not offset:
            maps_dict[dict_key] = {
                "input": group_dict,
                "reference_group": reference_dict,
            }

    return maps_dict


def merge_strds_groups(list_of_map_rows):
    """Merge a list of lists with grouped STRDS representations
    created with get_registered_maps into a single list

    This function concatenates map IDs and semantic labels of elemnts in
    map_rows_a and map_rows_b, matched by equal time stamp. Elements without
    match in both groups are excluded from the results. Thus, a precondition
    is that maps have equal time stamps and the evtl. where-clause used to
    select from both input STRDS is applicable both.

    Dev note: The presence of required semantic labels should be checked
              for each ganule at a later point!

    If no matching elements between input STRDS is found, the module exits.

    The purpose of the function is to be able to use multiple input STRDS
    (input and reference_strds_option), e.g. if original and artificial bands
    (spectral indices) are organized in different STRDS that need to be input
    to the deep learning models.
    """
    if len(list_of_map_rows) == 1:
        return list_of_map_rows[0]

    # Initialize a dict with granule as keys
    map_rows_dict = {map_row[0]: map_row[1] for map_row in list_of_map_rows[0]}
    # Merge all other map lists
    for map_rows_list in list_of_map_rows[1:]:
        for map_row in map_rows_list:
            if map_row[0] in map_rows_dict:
                map_rows_dict[map_row[0]]["semantic_labels"] += (
                    "," + map_row[1]["semantic_labels"]
                )
                map_rows_dict[map_row[0]]["ids"] += "," + map_row[1]["ids"]
    return map_rows_dict.items()


def compile_image_groups(
    input_strds,
    where=None,
    reference_strds=None,
    reference_where=None,
    sampling=None,
    spatial_relation=None,
    offset=None,
    dbif=None,
):
    """Compile dictionary with image groups as input to
    i.pytorch.predict

    starting with input
    strds and then add raster maps from a reference group that
    is either
    a)  sampled from a reference STRDS (if given) whose maps
        are related to the raster maps in the input STRDS in
        space (spatial_relation) and time (sampling) as
        requested or
    b) taken from the input_strds with the requested offset
       (if given)

    All raster maps are expected to have a semantic label
    associated.

    Currently supported use-cases are:
    a) only input STRDS (usually grouped ("one process per scene"))
    b) only input STRDS (usually grouped) with reference defined by
      offset (e.g. for "repeat-pass")
    c) input STRDS and reference STRDS matched according to temporal relation
       with grouped semantic labels

    Returns a dict of matched and grouped raster maps with
    the following structure:
    {temporal_key:
      {"input": {"semantic_label": "raster_map_id"}}
      {"reference_strds": {"semantic_label": "raster_map_id"}}}
    or
    {temporal_key:
      {"input": {"semantic_label": "raster_map_id"}}
      {"reference_strds": {}}}
    Where the temporal key can be either a temporal extent as tuple
    (start_time, end_time) or a tgis.RasterDataset with a TemporalExtent
    """
    time_unit = input_strds.get_relative_time_unit()

    spatial_extent = None
    # spatial_topology = None
    if spatial_relation:
        spatial_extent = gs.parse_command("g.region", flags="ug")
        # spatial_topology = (
        #     "3d"
        #     if input_strds.spatial_extent.top
        #     and reference_strds
        #     and reference_strds.spatial_extent.top
        #     else "2d"
        # )

    # If needed, here we could introduce multiple input STRDS
    # e.g. if original and artificial bands (spectral indices)
    # are organized in different STRDS
    # could be done with a merge_strds_groups() function
    # that concatenates map IDs and semantic labels
    # precondition is that maps have equal time stamps and the evtl.
    # where-clause is applicable to all STRDS
    map_rows = get_registered_maps(
        input_strds,
        columns=["id", "start_time", "end_time", "semantic_label"],
        where=where,
        group_by=["start_time", "end_time"],
        spatial_extent=spatial_extent,
        spatial_relation=spatial_relation,
        dbif=dbif,
    )

    if not map_rows:
        gs.warning(_("No data selected from STRDS <{}>").format(options["input"]))
        sys.exit(0)

    if reference_strds:
        # Check user input
        if not sampling:
            gs.fatal(_("Sampling is required to match reference image groups."))
        if offset:
            gs.warning(_("'reference_strds' is given. Offset will be ignored."))

        map_rows_reference = get_registered_maps(
            reference_strds,
            columns=["id", "start_time", "end_time", "semantic_label"],
            where=reference_where,
            group_by=["start_time", "end_time"],
            spatial_extent=spatial_extent,
            spatial_relation=spatial_relation,
            dbif=dbif,
        )

        map_rows = build_group_dict(map_rows, time_unit=time_unit, raster_dataset=True)
        map_rows_reference = build_group_dict(
            map_rows_reference, time_unit=time_unit, raster_dataset=True
        )

        topo_builder = tgis.SpatioTemporalTopologyBuilder()
        topo_builder.build(
            mapsA=list(map_rows.keys()),
            mapsB=list(map_rows_reference.keys()),
            # spatial=spatial_topology,
        )

        map_groups = {}
        for map_dataset, group_elements in map_rows.items():
            matched_reference = False
            for topology in sampling:
                matching_objects = getattr(map_dataset, topology)
                # Check if any maps are temporally related to the granule with the given temporal topology
                if matching_objects:
                    matched_reference = True
                    for matching_object in matching_objects:
                        # Get elements frm matched reference group
                        matched_group_elements = map_rows_reference[matching_object][
                            "input"
                        ]
                        group_elements["reference_group"].update(matched_group_elements)

            if not matched_reference:
                gs.warning("No matching objects")
                continue
            group_dict = {map_dataset.get_temporal_extent_as_tuple(): group_elements}
            map_groups.update(group_dict)
        return map_groups

    return build_group_dict(map_rows, time_unit=time_unit, offset=offset)


def distribute_cores(nprocs, groups_n):
    """Distribute cores across inner (parallel processes within
    i.pytorch.predict) and outer (parallel runs of i.pytorch.predict)
    loop of processes. At least one core is allocated to inner
    (i.pytorch.predict) and outer (imagery group)
    process.
    Order of returns is inner, outer."""
    return max(1, floor(nprocs / groups_n)), min(groups_n, nprocs)


def process_scene_group(
    temporal_extent,
    map_dict,
    basename=None,
    module_options=None,
    torch_flags=None,
    skip_incomplete=False,
):
    """Create an imagery group from semantic labels of a temporal extent and
    run a pytorch prediction on the imagery group"""
    # Get the base name
    if not basename:
        output_name = os.path.commonprefix(list(map_dict["input"].values())).rstrip("_")
    else:
        output_name = f"{basename}_{temporal_extent[0].strftime('%Y%m%dT%H%M%S')}_{temporal_extent[1].strftime('%Y%m%dT%H%M%S')}"
    output_name = output_name.split("@", 1)[0]

    # Get semantic labels
    output_bands = json.loads(
        Path(module_options["configuration"]).read_text(encoding="UTF8")
    )["output_bands"]

    gs.verbose(_("Processing group {}...").format(output_name))

    try:
        torch_mod = Module(
            "i.pytorch.predict",
            output=output_name,
            stdout_=PIPE,
            **module_options,
            flags=torch_flags,
            run_=False,
            stderr_=PIPE,
            quiet=True,
        )
        for group in ["input", "reference_group"]:
            if map_dict[group]:
                Module(
                    "i.group",
                    group=f"{TMP_NAME}_{group}_{output_name}",
                    input=list(map_dict[group].values()),
                    quiet=True,
                )
                gs.debug(f"Maps in {group}: {', '.join(map_dict[group].values())}")
                torch_mod.inputs[group].value = f"{TMP_NAME}_{group}_{output_name}"
        torch_mod.run()
        gs.debug(torch_mod.outputs.stderr)
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
    except Exception:
        if not skip_incomplete:
            gs.fatal(
                _(
                    "Failed to produce output for {output_name} with the following error message: {error}."
                ).format(
                    output_name=output_name,
                    error=torch_mod.outputs["stderr"].value.strip(),
                )
            )
            return None
        gs.warning(
            _(
                "Failed to produce output for {output_name} with the following error message: {error}."
            ).format(
                output_name=output_name, error=torch_mod.outputs["stderr"].value.strip()
            )
        )
        return None


def main():
    """Do the main work"""

    # Initialize TGIS
    tgis.init()

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

    # Initialize input STRDS
    input_strds = tgis.open_old_stds(options["input"], "strds")
    # Initialize reference STRDS if requested
    reference_strds = None
    if options["reference_strds"]:
        reference_strds = tgis.open_old_stds(options["reference_strds"], "strds")

    # Compile dicts for imagery groups per granule (and semantic label)

    if options["auxillary_group"]:
        # Check content of the static group with auxillary data
        # Could be organized in subgroups if orbit or tile info
        # would be needed to filter
        # Needed to be propagated to i.pytorch.predict in case
        group_to_dict(options["auxillary_group"], fill_semantic_label=False)

    imagery_groups = compile_image_groups(
        input_strds,
        where=options["where"],
        reference_strds=reference_strds,
        reference_where=options["reference_where"],
        sampling=options["sampling"],
        spatial_relation=options["region_relation"],
        offset=int(options["offset"]) if options["offset"] else None,
    )

    if not imagery_groups:
        gs.warning("Insufficient data found to process.")
        sys.exit(0)

    # Check wich device to use
    if flags["c"]:
        device = "cpu"
    elif not torch.cuda.is_available():
        device = "cpu"
        gs.warning(_("No GPU available. Will run on CPU"))
    else:
        device = "gpu"

    # Distribute cores
    nprocs_inner, nprocs_outer = int(options["nprocs"]), 1
    if nprocs_inner > 1 and device == "cpu":
        # Distribute cores across inner and outer processes
        # if module runs on CPU
        nprocs_inner, nprocs_outer = distribute_cores(nprocs_inner, len(imagery_groups))
        gs.verbose(
            _("Using {n_outer} outer and {n_inner} processes.").format(
                n_inner=nprocs_inner, n_outer=nprocs_outer
            )
        )
    # Collect basic module_options for i.pytorch.predict
    module_options = {
        option: (
            list(map(int, options[option].split(",")))
            if option == "tile_size"
            else options[option]
        )
        for option in [
            "auxillary_group",
            "model",
            "model_code",
            "vector_tiles",
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
        skip_incomplete=flags["s"],
    )

    # Run predictions and collect
    if nprocs_outer > 1:
        with Pool(nprocs_outer) as pool:
            register_strings = pool.starmap(i_pytorch_predict, imagery_groups.items())
    else:
        register_strings = list(starmap(i_pytorch_predict, imagery_groups.items()))

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
    tmp_file = Path(gs.tempfile(create=False))
    tmp_file.write_text(
        "\n".join(
            register_string for register_string in register_strings if register_string
        )
        + "\n",
        encoding="UTF8",
    )

    # Register results in output STRDS
    register_maps_in_space_time_dataset(
        "raster",
        strds_long_name,
        file=tmp_file,
        update_cmd_list=False,
        fs="|",
    )

    # Remove tempfile
    tmp_file.unlink()


if __name__ == "__main__":
    options, flags = gs.parser()

    # Lazy imports
    try:
        import torch
    except ImportError:
        gs.fatal("Could not import pytorch. Please make sure it is installed")

    sys.exit(main())
