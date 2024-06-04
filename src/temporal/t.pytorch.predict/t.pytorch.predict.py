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

              ToDo:
                - Add a reference_strds option
                - add a reference_strds_where option
                - add a sampling option for spation-temporal matching of input and refgerence STRDS
                  using time and semantic label e.g. time = time AND semanitc_label = semantic_label
                - add the possibility to have "pattern" in semantic_label definition in model JSON

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

# %option G_OPT_STRDS_INPUT
# %key: reference_strds
# %end

# %option G_OPT_T_WHERE
# %key: reference_where
# % description: Where clause to select reference images
# %end

# %option G_OPT_I_GROUP
# %key: static_group
# % required: no
# % multiple: no
# % description: Input imagery group with time independent raster maps
# %end

# # %option
# # % key: region_relation
# # % type: string
# # % required: no
# # % multiple: no
# # %end

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
# % key: c
# % description: Use CPU as device for prediction, default is use cuda (GPU) if detected
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# %key: l
# % description: Limit output to valid range (data outside the valid range is set to valid min/max)
# %end

# %flag
# %key: s
# % description: Process each semantic label separately
# %end

# %rules
# % collective: title,description
# % required: -e,title,description
# %end

import os
import json
import sys

from functools import partial
from math import floor
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs


TMP_NAME = gs.tempname(12)
# Get GRASS GIS environment
GISENV = dict(gs.gisenv())


def parse_group(imagery_group):
    """Create a dict to represent an imagery group, where raster maps
    in the imagery group are the values and the associated semantic_labels
    are their respective keys.
    For raster maps in the imagery group that do not have a semantic label
    a warning is given.
    """
    group_dict = {}
    if not imagery_group:
        return group_dict
    maps_in_group = (
        gs.read_command("i.group", group=imagery_group, flags="g", quiet=True)
        .strip()
        .split()
    )

    for idx, raster_map, raster_map in enumerate(maps_in_group):
        raster_map_info = gs.raster_info(raster_map)
        semantic_label = raster_map_info["semantic_label"]
        if not raster_map_info["semantic_label"]:
            gs.warning(
                _(
                    "Raster map {rmap} in group {igroup} does not have a semantic label."
                    "Using the numeric index in the group"
                ).format(rmap=raster_map, igroup=imagery_group)
            )
            semantic_label = idx + 1
        group_dict[semantic_label] = raster_map
    return group_dict


def is_int(string):
    try:
        int(string)
    except Exception:
        return False
    return True


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
                _("Unable to get map ids from register table " "<{}>").format(
                    stds.get_map_register()
                )
            )
            raise

    if connection_state_changed:
        dbif.close()

    return rows


def build_group_dict(
    map_list,
    time_unit=None,
    offset=None,
    raster_dataset=False,
    reference_suffix=None,
    static_map_dict=None,
):
    """Build dictionary to represent image groups registered in an STRDS"""
    maps_dict = {}
    for idx, raster_map_row in enumerate(map_list):
        reference_dict = {}
        temporal_extent = [raster_map_row["start_time"], raster_map_row["end_time"]]
        if offset and 0 < idx + offset < len(map_list):
            temporal_extent.extend(
                [
                    map_list[idx + offset]["start_time"],
                    map_list[idx + offset]["end_time"],
                ]
            )
            map_ids = raster_map_row["ids"].split(",")
            if not map_list[idx + offset]["semantic_labels"]:
                gs.warning(
                    "Semantic labels are missing for all raster maps in reference"
                )
                semantic_labels = [""] * len(map_ids)
            else:
                semantic_labels = map_list[idx + offset]["semantic_labels"].split(",")
                if reference_suffix:
                    semantic_labels = [
                        f"{semantic_label}_{reference_suffix}"
                        for semantic_label in semantic_labels
                    ]
            reference_dict = {
                semantic_labels[mid]: map_id for mid, map_id in enumerate(map_ids)
            }
        temporal_extent = {time_stamp for time_stamp in temporal_extent if time_stamp}
        temporal_extent = (
            min(temporal_extent),
            max(temporal_extent) if len(temporal_extent) > 1 else None,
        )
        if raster_dataset:
            dict_key = tgis.RasterDataset(None)
            if time_unit:
                dict_key.set_relative_time(*temporal_extent, time_unit)
            else:
                dict_key.set_absolute_time(*temporal_extent)
        else:
            dict_key = temporal_extent

        map_ids = raster_map_row["ids"].split(",")
        if not raster_map_row["semantic_labels"]:
            gs.warning("Semantic labels are missing for all raster maps")
            semantic_labels = [""] * len(map_ids)
        else:
            semantic_labels = raster_map_row["semantic_labels"].split(",")
        group_dict = {
            semantic_labels[idx]: map_id for idx, map_id in enumerate(map_ids)
        }
        if static_map_dict:
            group_dict.update(static_map_dict)
        if not offset or offset and reference_dict:
            group_dict.update(reference_dict)
            maps_dict[dict_key] = group_dict
    return maps_dict


def compile_image_groups(
    input_strds,
    where=None,
    reference_strds=None,
    reference_where=None,
    reference_suffix=None,
    static_group=None,
    sampling=None,
    spatial_relation=None,
    offset=None,
    dbif=None,
):
    """Compile image groups starting with input strds and then
    add:
    a) raster maps from the static input group
    b) reference raster maps that are either
        i)  sampled from a reference STRDS (if given) whose maps
            are related to the raster maps in the input STRDS in
            space (spatial_relation) and time (sampling) as
            requested or
        ii) taken from the input_strds with the requested offset

    All raster maps are expected to have a semantic label
    associated. If semantic labels are supposed to be matched
    between currnt and reerence groups this function should be used
    in a loop over semantic_labels.
    Cases are:
    a) just input STRDS (usually grouped ("one process per scene"))
    b) just input STRDS (usually grouped) with reference defined by
      offset (e.g. for "repeat-pass" with multiple semantic labels per scene)
    c) just input STRDS (usually grouped) with reference defined by
      offset (e.g. for "repeat-pass" with multiple semantic labels per scene)
    d) input STRDS and reference STRDS matched according to temporal relation
       with grouped semantic labels
    e) input STRDS and reference STRDS matched according to temporal relation
       with equal semantic labels

    - input raster maps and reference raster maps
    Returns a dict of matched and grouped raster maps"""
    time_unit = input_strds.get_relative_time_unit()
    static_maps = {}
    if static_group:
        static_maps = parse_group(static_group)

    spatial_extent = None
    spatial_topology = None
    if spatial_relation:
        spatial_extent = gs.parse_command("g.region", flags="ug")
        spatial_topology = (
            "3d"
            if input_strds.spatial_extent.top
            and reference_strds
            and reference_strds.spatial_extent.top
            else "2d"
        )

    map_rows = get_registered_maps_grouped(
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
        # Case d and e
        map_rows_reference = get_registered_maps_grouped(
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
            spatial=spatial_topology,
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
                        # for ref_dict in map_rows_reference[matching_object].values():
                        # print(ref_dict)
                        matched_group_elements = map_rows_reference[matching_object]
                        if reference_suffix:
                            matched_group_elements = {
                                f"{key}_{reference_suffix}": val
                                for key, val in map_rows_reference[
                                    matching_object
                                ].items()
                            }
                        group_elements.update(matched_group_elements)

            if not matched_reference:
                gs.warning("No matching objects")
                continue
            group_dict = {map_dataset.get_temporal_extent_as_tuple(): group_elements}
            map_groups.update(group_dict)
        return map_groups

    # Group maps using granule
    return build_group_dict(
        map_rows,
        time_unit=time_unit,
        offset=offset,
        reference_suffix=reference_suffix,
        static_map_dict=static_maps,
    )


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

    try:
        Module(
            "i.pytorch.predict",
            input=f"{TMP_NAME}_{output_name}",
            output=output_name,
            stdout_=PIPE,
            **module_options,
            flags=torch_flags,
            # stderr_=PIPE,
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
    except RuntimeError as error:
        gs.fatal(
            _(
                "Failed to produce output for {output_name} with the following error message: {error}."
            ).format(output_name=output_name, error=error)
        )
        return None


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

    # Initialize input STRDS
    input_strds = tgis.open_old_stds(options["input"], "strds", dbif)
    # Initialize reference STRDS if requested
    reference_strds = None
    if options["reference"]:
        reference_strds = tgis.open_old_stds(options["input"], "strds", dbif)

    # Compile dicts for imagery groups per granule (and semantic label)
    static_maps = parse_group(options["static_group"])
    imagery_groups = compile_image_groups(
        input_strds,
        where=options["where"],
        reference_strds=reference_strds,
        reference_where=options["reference_where"],
        reference_suffix=options["reference_suffix"],
        static_group=static_maps,
        sampling=options["sampling"],
        spatial_relation=options["spatial_relation"],
        offset=options["spatial_relation"],
        dbif=dbif,
    )

    if not imagery_groups:
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
            register_strings = pool.starmap(i_pytorch_predict, imagery_groups.items())
    else:
        register_strings = [
            i_pytorch_predict(*scene_group) for scene_group in imagery_groups.items()
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
    tmp_file = Path(gs.tempfile(create=False))
    tmp_file.write_text("\n".join(register_strings) + "\n", encoding="UTF8")

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

    import grass.temporal as tgis
    from grass.pygrass.modules.interface import Module
    from grass.temporal.register import register_maps_in_space_time_dataset

    sys.exit(main())
