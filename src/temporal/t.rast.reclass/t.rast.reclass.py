#!/usr/bin/env python3

############################################################################
#
# MODULE:       t.rast.reclass
# AUTHOR(S):    Stefan Blumentrath
#
# PURPOSE:      Reclassify maps in a SpaceTimeRasterDataset
# COPYRIGHT:    (C) 2022 by the Stefan Blumentrath and
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
# % description: Reclassify maps in a SpaceTimeRasterDataset.
# % keyword: temporal
# % keyword: reclassification
# % keyword: raster
# % keyword: time
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_STRDS_OUTPUT
# %end

# %option
# % key: rules
# % type: string
# % required: yes
# % multiple: no
# % key_desc: name
# % label: File containing reclass rules
# % description: '-' for standard input
# % gisprompt: old,file,file
# %end

# %option
# % key: semantic_label
# % type: string
# % required: no
# % multiple: no
# % description: Semantic label to be assigned to the reclassified map - also used as a suffix that is appended to input map name
# %end

# %option
# % key: temporaltype
# % type: string
# % required: no
# % multiple: no
# % options: absolute,relative
# % key_desc: name
# % description: The temporal type of the space time dataset
# %end

# %option
# % key: semantictype
# % type: string
# % required: no
# % multiple: no
# % options: min,max,sum,mean
# % description: Semantic type of the space time dataset
# %end

# %option
# % key: title
# % type: string
# % required: no
# % multiple: no
# % description: Title of the new space time dataset
# %end

# %option
# % key: description
# % type: string
# % required: no
# % multiple: no
# % description: Description of the new space time dataset
# %end

# %option G_OPT_M_NPROCS
# % key: nprocs
# % type: integer
# % description: Number of r.reclass processes to run in parallel
# % required: no
# % multiple: no
# % answer: 1
# %end

# %option G_OPT_T_WHERE
# %end

# %flag
# % key: n
# % description: Register Null maps
# %end

# %flag
# % key: e
# % description: Extend existing space time raster dataset
# %end

# %rules
# % required: -e, title
# % excludes: -e, title, description
# % collective: title, description
# %end

# ToDo:
# - add semantic label support to register_map_object_list
#   https://grass.osgeo.org/grass83/manuals/libpython/_modules/temporal/register.html#register_maps_in_space_time_dataset

import sys

from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import grass.script as gs
from grass.pygrass.modules import Module


############################################################################


def run_reclassification(
    reclass_module, input_tuple, register_null, semantic_label, current_mapset
):
    """Run the pygrass modules with input and return RasterDataset
    with reclassified map
    :param reclass_module: A PyGRASS Module object with a pre-configured
                           r.reclass module
    :param input_tuple: A tuple containg the full map name, start-time and
                        end-time of the map
    :param register_null: Boolean if maps with only NULL should be registered
                          in the output STRDS (default: True)
    :param semantic_label: Semantic label to assign to the output raster maps
    :param current_mapset: Name of the current mapset
    :return: A string for registering the reclassified map in the output
             SpaceTimeRasterDataset or None
    """
    input_name = input_tuple[0].split("@")[0]
    output_name = f"{input_name}_{semantic_label or 'rc'}"

    if (
        gs.find_file(name=output_name, mapset=current_mapset, element="cell")
        and gs.overwrite() is False
    ):
        gs.fatal(
            _(
                "Unable to perform reclassification. Output raster "
                "map <{name}> exists and overwrite flag was not set"
            ).format(name=output_name)
        )
    reclass_module.inputs.input = input_name
    reclass_module.outputs.output = output_name
    reclass_module.run()

    # In case of a null map continue, remove it and return None to register
    if not register_null:
        # Compute statistics
        gs.run_command("r.support", flags="s", map=output_name)
        new_map_info = gs.raster_info(output_name)
        if new_map_info["min"] is None and new_map_info["max"] is None:
            gs.run_command("g.remove", flags="f", type="raster", name=output_name)
            return None

    if semantic_label and input_tuple[2]:
        return f"{output_name}|{input_tuple[1]}|{input_tuple[2]}|{semantic_label}"
    if semantic_label and not input_tuple[2]:
        return f"{output_name}|{input_tuple[1]}|{semantic_label}"
    if not semantic_label and not input_tuple[2]:
        return f"{output_name}|{input_tuple[1]}"
    if not semantic_label and input_tuple[2]:
        return f"{output_name}|{input_tuple[1]}|{input_tuple[2]}"


def reclass_temporal_map(
    map_list,
    reclass_module,
    register_null,
    semantic_label,
    nprocs=1,
    overwrite=False,
):
    """Reclassify a list of raster input maps with r.reclass
    This is mainly a wrapper to parallelize the run_reclassification function
    :param map_list: A list of RasterDataset objects that contain the raster
                     maps that should be reclassified
    :param reclass_module: A PyGRASS Module object with a pre-configured
                           r.reclass module
    :param register_null: Boolean if maps with only NULL should be registered
                          in the output STRDS (default: True)
    :param semantic_label: Semantic label to assign to the output raster maps
    :param nprocs: The number of processes used for parallel computation
    :param overwrite: Overwrite existing raster maps
    :return: A list of strings for registering reclassified maps in the output
             SpaceTimeRasterDataset
    """
    current_mapset = gs.gisenv()["MAPSET"]
    nprocs = min(nprocs, len(map_list))
    if nprocs > 1:
        with Pool(nprocs) as p:
            output_list = p.starmap(
                run_reclassification,
                [
                    (
                        deepcopy(reclass_module),
                        (
                            raster_map.get_id(),
                            *raster_map.get_temporal_extent_as_tuple(),
                        ),
                        register_null,
                        semantic_label,
                        current_mapset,
                    )
                    for raster_map in map_list
                ],
            )
    else:
        output_list = [
            run_reclassification(
                deepcopy(reclass_module),
                (
                    raster_map.get_id(),
                    *raster_map.get_temporal_extent_as_tuple(),
                ),
                register_null,
                semantic_label,
                current_mapset,
            )
            for raster_map in map_list
        ]

    return output_list


def main():
    # Get the options
    input = options["input"]
    output = options["output"]
    where = options["where"]
    # base = options["basename"]
    register_null = flags["n"]
    nprocs = int(options["nprocs"])

    # Initialize TGIS
    tgis.init()

    # Connect to TGIS DB
    dbif = tgis.SQLDatabaseInterfaceConnection()
    dbif.connect()

    # Open input STRDS
    sp = tgis.open_old_stds(input, "strds", dbif)

    map_list = sp.get_registered_maps_as_objects(
        where=where, order="start_time", dbif=dbif
    )

    if not map_list:
        dbif.close()
        gs.fatal(_("Space time raster dataset <{}> is empty".format(input)))

    # We will create the strds later, but need to check here
    tgis.check_new_stds(output, "strds", dbif, gs.overwrite())

    # Create Module object that will be deep copied
    # and be put into the process queue
    reclass_module = Module(
        "r.reclass",
        overwrite=gs.overwrite,
        quiet=True,
        run_=False,
    )

    # Get reclassification rules
    reclass_rules = options["rules"]
    if reclass_rules == "-":
        reclass_module.inputs.rules = "-"
        reclass_str = str(sys.__stdin__.read())
        if not reclass_str:
            gs.fatal(_("Empty or no reclass rules provided"))
        elif "=" not in reclass_str:
            gs.fatal(_("Invalid reclass rules provided"))
        reclass_module.inputs["stdin"].value = reclass_str
    elif not Path(reclass_rules).exists():
        gs.fatal(_("Invalid reclass rules provided"))
    else:
        reclass_module.inputs.rules = reclass_rules

    # Run reclassification
    output_list = reclass_temporal_map(
        map_list,
        reclass_module,
        register_null,
        options["semantic_label"],
        nprocs=nprocs,
        overwrite=gs.overwrite(),
    )

    # Register produced maps
    if output_list:
        # Create new or overwrite existing
        output_strds = tgis.factory.dataset_factory(
            "strds", f"{output}@{gs.gisenv()['MAPSET']}"
        )
        if not output_strds.is_in_db(dbif) or (gs.overwrite() and not flags["e"]):
            # Get basic metadata
            temporal_type, semantic_type, title, description = sp.get_initial_values()

            # Create new STRDS
            output_strds = tgis.open_new_stds(
                output,
                "strds",
                options["temporaltype"] or temporal_type,
                options["title"] or title,
                options["description"] or description,
                options["semantictype"] or semantic_type,
                dbif,
                gs.overwrite(),
            )

        # Append to existing
        elif output_strds.is_in_db(dbif) and flags["e"]:
            output_strds = tgis.open_old_stds(output, "strds", dbif)

        # Register reclassified maps
        register_file_path = gs.tempfile(create=False)
        with open(register_file_path, "w") as register_file:
            register_file.write("\n".join([ol for ol in output_list if ol]))
        tgis.register_maps_in_space_time_dataset(
            "raster",
            output,
            file=register_file_path,
            dbif=dbif,
        )

    else:
        gs.warning(_("No output maps to register"))

    dbif.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    import grass.temporal as tgis

    main()
