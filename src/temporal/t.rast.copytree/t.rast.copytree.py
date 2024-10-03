#! /usr/bin/python3
"""
MODULE:    t.rast.copytree
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Transfer raster map files from STRDS in external GDAL format to target directory
COPYRIGHT: (C) 2024 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General
Public License (>=v2). Read the file COPYING that
comes with GRASS for details.
"""

# %module
# % description: Transfer raster map files from STRDS in external GDAL format to target directory
# % keyword: temporal
# % keyword: move
# % keyword: copy
# % keyword: GDAL
# % keyword: directory
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_M_DIR
# % key: output_directory
# % description: Path to the output / destination directory
# % required: yes
# %end

# %option G_OPT_M_DIR
# % key: source_directory
# % guisection: output
# % description: Path to the source directory
# % required: no
# %end

# %option
# % key: suffix
# % type: string
# % description: Suffix of files to transfer
# % required: no
# %end

# %option
# % key: temporal_tree
# % type: string
# % description: Strftime format to create temporal directory name or tree (e.g. "%Y/%m/%d")
# % required: no
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: m
# % label: Move files into destination (default is copy)
# % description: Move files into destination (default is copy)
# %end

# %flag
# % key: o
# % label: Overwrite existing files
# % description: Overwrite existing files
# %end

# %flag
# % key: s
# % label: Use semantic label in directory structure
# % description: Use semantic label in directory structure
# %end

# %rules
# % collective: source_directory, suffix
# %end

import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import grass.script as gs
import grass.temporal as tgis

OVERWRITE = False


def _move(source, target):
    """Wrapper for shutil.move"""
    target = Path(target)
    if target.exists():
        if not OVERWRITE:
            gs.fatal(
                _("Target <{}> exists. Please use the overvrite flag.").format(
                    str(target)
                )
            )
        target.unlink()

    shutil.move(source, target)


def _copy(source, target):
    """Wrapper for shutil.copy2"""
    target = Path(target)
    if target.exists() and not OVERWRITE:
        gs.fatal(
            _("Target <{}> exists. Please use the overvrite flag.").format(str(target))
        )
    shutil.copy2(source, target)


def transfer_results(
    raster_map_rows,
    source_directory,
    output_directory=None,
    temporal_tree="%Y/%m/%d",
    sep="|",
    suffix="tif",
    use_semantic_label=False,
    transfer_function=_copy,
    nprocs=1,
):
    """Transfer resulting maps to Network storage"""
    # Set relevant time frame
    target_directories = set()
    transfer_tuples = []
    return_list = []
    for map_row in raster_map_rows:
        start_day = map_row["start_time"]
        if use_semantic_label:
            semantic_label = map_row["semantic_label"]
            target_directory = (
                output_directory / semantic_label / start_day.strftime(temporal_tree)
            )
        else:
            target_directory = output_directory / start_day.strftime(temporal_tree)
        target_directories.add(target_directory)
        transfer_tuples.append(
            (
                f"{source_directory / map_row['name']!s}{suffix}",
                f"{target_directory / map_row['name']!s}{suffix}",
            )
        )
        return_list.append(
            sep.join(
                [
                    map_row["name"],
                    map_row["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    map_row["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    map_row["semantic_label"] or "",
                    f"{target_directory / map_row['name']!s}{suffix}",
                ]
            )
        )

    # Create target directory structure
    for target_directory in target_directories:
        target_directory.mkdir(exist_ok=True, parents=True)

    # Transfer files in parallel
    if nprocs > 1:
        with Pool(nprocs) as pool:
            pool.starmap(transfer_function, transfer_tuples)
    else:
        for transfer_tuple in transfer_tuples:
            transfer_function(*transfer_tuple)
    return return_list


def main():
    """Do the main work"""
    options, flags = gs.parser()
    global OVERWRITE
    OVERWRITE = flags["o"]

    # Check if maps are exported to GDAL formats
    if options["source_directory"]:
        source_directory = Path(options["source_directory"])
        suffix = (
            f".{options['suffix']}"
            if not options["suffix"].startswith(".")
            else options["suffix"]
        )
    else:
        external = {
            line.split(": ")[0]: line.split(": ")[1]
            for line in gs.read_command("r.external.out", flags="p").split("\n")
            if ": " in line
        }
        if not external:
            gs.warning(
                _(
                    "Neither source directory given nor external linking (r.external.out) defined."
                )
            )
            sys.exit(0)
        source_directory = Path(external["directory"])
        suffix = f"{external['extension']}" if external["extension"] != "<none>" else ""

    tgis.init()
    input_strds = tgis.open_old_stds(options["input"], "strds")
    input_strds_maps = input_strds.get_registered_maps(
        columns="name,start_time,end_time,semantic_label", where=options["where"]
    )

    register_strings = transfer_results(
        input_strds_maps,
        source_directory,
        output_directory=Path(options["output_directory"]),
        temporal_tree=options["temporal_tree"] or "%Y/%m/%d",
        sep="|",
        suffix=suffix,
        use_semantic_label=flags["s"],
        transfer_function=_move if flags["m"] else _copy,
        nprocs=1,
    )

    # Print register information
    print("\n".join(register_strings))


if __name__ == "__main__":
    sys.exit(main())
