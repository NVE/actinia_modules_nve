#!/usr/bin/env python3

"""MODULE:   t.sentinel3.import
AUTHOR(S):   Stefan Blumentrath
PURPOSE:     Import and pre-process Sentinel-3 data from the Copernicus program
             into a Space Time Raster Dataset (STRDS)
COPYRIGHT:   (C) 2024-2025 by Norwegian Water and Energy Directorate,
             Stefan Blumentrath, and the GRASS development team

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
# % required: no
# % description: Anxillary data bands to import (e.g. LST_uncertainty, default is None, use "all" to import all available)
# %end

# %option
# % key: flag_bands
# % multiple: yes
# % required: no
# % description: Quality flag bands to import (e.g. bayes_in, default is None, use "all" to import all available)
# %end

# %option
# % key: maximum_solar_angle
# % type: double
# % description: Import only pixels where solar angle is lower or equal to the given maximum
# % required: no
# %end

# %option
# % key: swath_mask_band
# % type: string
# % description: Band to use for creating a swath mask
# % required: no
# %end

# %option
# % key: swath_mask_buffer
# % type: integer
# % description: Create a swath mask for removing border noise by buffering the swath_mask_band inwards (in number of pixels)
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
# % collective: swath_mask_band,swath_mask_buffer
# % required: -e,title,description
# %end

import re
import sys
from datetime import datetime, timezone
from functools import partial
from math import floor
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs

S3_FILE_PATTERN = {
    # "S3OL1ERF": None,
    "S3SL1RBT": "**/S3*SL_1_RBT__*.zip",
    "S3SL2LST": "**/S3*SL_2_LST__*.zip",
}


def check_file_input(file_input: str, product_type: str) -> list[Path]:
    """Check input for files to geocode.

    If input is a directory (and not a SAFE), contained SAFE files are listed
    If input is a text file each line is assumed to be a path to a SAFE file
    If input is a comma separated list of files element is assumed to be a
    path to a SAFE file.
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


def check_swath_mask_input(module_options: dict) -> tuple[str, int] | None:
    """Check input for producing swath masks.

    Returns parsed dict values if requested, None if not,
    and raises errors for invalid input.
    """
    if not module_options["swath_mask_band"]:
        return None, None

    swath_mask_band = module_options["swath_mask_band"]
    try:
        swath_mask_buffer = int(module_options["swath_mask_buffer"])
    except ValueError:
        gs.fatal(_("Buffer for swath_mask_buffer must be an integer."))

    if not any(
        swath_mask_band.replace("reflectance", "radiance") in module_options[bands]
        or swath_mask_band in module_options[bands]
        or module_options[bands] == "all"
        for bands in ("bands", "anxillary_bands", "flag_bands")
    ):
        gs.fatal(
            _(
                "Band {} to be used as swath mask is not requested for import in bands, anxillary_bands or flag_bands",
            ).format(swath_mask_band),
        )
    del module_options["swath_mask_band"]
    del module_options["swath_mask_buffer"]

    return swath_mask_band, swath_mask_buffer


def check_files_list(file_path_list: str | None, product_type: str) -> list[Path]:
    """Check if files in a list of files exist and gives a warning otherwise."""
    if not file_path_list:
        gs.fatal(_("No scenes found to process"))
    existing_paths = []
    file_pattern = re.compile(
        f".*{S3_FILE_PATTERN[product_type].replace('*', '.*')}",
        re.IGNORECASE,
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
                        "File {file_path} does not match expected pattern {pattern} of product {product_type}.",
                    ).format(
                        file_path=file_path,
                        pattern=S3_FILE_PATTERN[product_type],
                        product_type=product_type,
                    ),
                )
        else:
            gs.warning(_("File {} not found").format(file_path))
    return existing_paths


def parse_s3_file_name(file_name: str) -> dict:
    """Extract info from Sentinel-3 file name according to naming convention.

    Naming convention is described here:
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-3-slstr/naming-convention
    Assumes that file name is checked to be a valid / supported Sentinel-3 file name.
    E.g.:
    "S3B_SL_1_RBT____20240129T110139_20240129T110439_20240130T114811_0180_089_094_1800_PS2_O_NT_004.SEN3"
    The suffix does not matter.

    :param file_name: string representing the file name of a Senintel-3 scene
    """
    try:
        return {
            "mission_id": file_name[0:3],
            "instrument": file_name[4:6],
            "level": file_name[7],
            "product": file_name[9:12],
            "start_time": datetime.strptime(file_name[16:31], "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc,
            ),
            "end_time": datetime.strptime(file_name[32:47], "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc,
            ),
            "ingestion_time": datetime.strptime(
                file_name[48:63],
                "%Y%m%dT%H%M%S",
            ).replace(tzinfo=timezone.utc),
            "duration": file_name[64:68],
            "cycle": file_name[69:72],
            "relative_orbit": file_name[73:76],
            "frame": file_name[77:81],
        }
    except ValueError:
        gs.fatal(_("{} is not a supported Sentinel-3 scene").format(file_name))


def group_scenes(
    s3_files: list[Path],
    group_variables: tuple[str] = (
        "mission_id",
        "instrument",
        "level",
        "product",
        "cycle",
        "relative_orbit",
    ),
) -> dict:
    """Group scenes along track and date by information from the file name.

    The following variables can be used to group the scenes:
     1. mission ID
     2. instrument
     3. level
     4. product
     5. cycle
     6. relative orbit
     7. temporal granule

    :param s3_files: list of pathlib.Path objects with Sentinel-3 files
    :param group_variables: tuple of strings with the variables to group the scenes by
    """
    groups = {}
    for sfile in s3_files:
        s3_name_dict = parse_s3_file_name(sfile.name)
        s3_name_dict["level"] = (
            "2" if s3_name_dict["level"] == "1" else s3_name_dict["level"]
        )
        group_id = "_".join(
            [s3_name_dict[group_var] for group_var in group_variables]
            + [s3_name_dict["start_time"].strftime("%Y%m%d")],
        )
        if group_id in groups:
            groups[group_id].append(str(sfile))
        else:
            groups[group_id] = [str(sfile)]
    return groups


def process_scene_group(
    scene_group: tuple,
    module_options: dict | None = None,
    module_flags: str | None = None,
    swath_mask: tuple[str, int] | None = None,
) -> str:
    """Import a group of Sentinel3 scenes and create a swath mask if requested.

    :param scene_group: tuple with group name and list of scene file paths
    :param module_options: dictionary with module options
    :param module_flags: list of module flags
    """
    gs.verbose(_("Processing scene group {}...").format(scene_group[0]))

    module_options["basename"] = module_options["basename"] or scene_group[0]

    swath_mask_band, swath_mask_buffer = swath_mask

    i_sentinel3_import = Module(
        "i.sentinel3.import",
        input=",".join(scene_group[1]),
        stdout_=PIPE,
        **module_options,
        flags=module_flags,
        quiet=True,
    )
    module_stdout = i_sentinel3_import.outputs.stdout
    module_stdout = module_stdout.strip() if module_stdout else ""

    if swath_mask_band and module_stdout:
        if swath_mask_band not in module_stdout:
            gs.warning(
                _("{band} not imported for group {group}").format(
                    band=swath_mask_band,
                    group=scene_group[0],
                ),
            )
        else:
            swath_mask_band_map, start_time, end_time, semantic_label = [
                line.split("|")
                for line in module_stdout.split("\n")
                if swath_mask_band in line
            ][0]
            swath_map_name = f"{module_options['basename']}_swath_mask"
            Module(
                "r.grow",
                input=swath_mask_band_map,
                output=swath_map_name,
                radius=-swath_mask_buffer,
                old=1,
                new=-1,
                quiet=True,
                env_=i_sentinel3_import.env_,
            )
            module_stdout += f"\n{swath_map_name}@{gs.gisenv().get('MAPSET')}|{start_time}|{end_time}|S3_swath_mask"

    return module_stdout


def distribute_cores(nprocs: int, groups_n: int) -> tuple[int, int]:
    """Distribute cores across expected processes.

    Distribute cores across inner (parallel processes within
    i.sentinel3.import) and outer (parallel runs of i.sentinel3.import)
    loop of processes. At least one core is allocated to inner
    (i.sentinel3.import) and outer (group of Sentinel-3 scenes)
    process.
    Order if returns is inner, outer.

    :param nprocs: number of available cores
    :param groups_n: number of groups to process
    """
    return max(1, floor(nprocs / groups_n)), min(groups_n, nprocs)


def main() -> None:
    """Do the main work."""
    # Get GRASS GIS environment
    gisenv = dict(gs.gisenv())

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{gisenv['MAPSET']}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if input is complete and valid
    # Check if target STRDS exists and create it if not
    # or abort if overwriting is not allowed
    if tgis_strds.is_in_db():
        if not gs.overwrite():
            gs.fatal(
                _(
                    "Output STRDS <{}> exists."
                    "Use --overwrite together with -e to modify the existing STRDS.",
                ).format(options["output"]),
            )
    elif not options["title"] or not options["description"]:
        gs.fatal(
            _(
                "Creation of a new STRDS <{}> requires the 'title' and 'description' option",
            ).format(strds_long_name),
        )

    # Group input scenes
    groups_to_process = group_scenes(
        check_file_input(options["input"], options["product_type"]),
    )

    # Distribute cores
    nprocs_inner, nprocs_outer = distribute_cores(
        int(options["nprocs"]),
        len(groups_to_process),
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
        swath_mask=check_swath_mask_input(options),
    )

    if nprocs_outer > 1:
        with Pool(nprocs_outer) as pool:
            register_strings = pool.map(import_module, groups_to_process.items())
    else:
        register_strings = [
            import_module(scene_group) for scene_group in groups_to_process.items()
        ]

    # Remove empty results
    register_strings = [result for result in register_strings if result]
    if not register_strings:
        gs.warning(
            _("No valid data found in <{}>. Nothing to register in STRDS.").format(
                options["input"],
            ),
        )
        sys.exit(0)

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{gisenv['MAPSET']}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if target STRDS exists and create it if not
    # or abort if overwriting is not allowed
    if tgis_strds.is_in_db() and not gs.overwrite():
        gs.fatal(
            _(
                "Output STRDS <{}> exists."
                "Use --overwrite together with -e to modify the existing STRDS.",
            ).format(options["output"]),
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
