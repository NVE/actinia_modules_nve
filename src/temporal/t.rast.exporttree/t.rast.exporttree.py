#! /usr/bin/python3
"""MODULE:    t.rast.exporttree
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Transfer raster map files from STRDS in external GDAL format to target directory
COPYRIGHT: (C) 2024 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General
Public License (>=v2). Read the file COPYING that
comes with GRASS for details.
"""

# %module
# % description: Export raster maps from a STRDS to GeoTiff format in a temporal directory structure.
# % keyword: temporal
# % keyword: move
# % keyword: export
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

# %option
# % key: format
# % type: string
# % description: Geotiff format flavor to export
# % options: COG,GTiff
# % answer: COG
# % required: no
# %end

# %option
# % key: compression
# % type: string
# % description: Compression method to be used for the output GeoTiffs
# % options: LZW,ZSTD,DEFLATE,LERC
# % answer: ZSTD
# % required: no
# %end

# %option
# % key: level
# % type: integer
# % description: Compression level for the output files (0-22), GDAL default for ZSTD is 9, for DEFLATE is 6
# % required: no
# %end

# %option
# % key: resampling
# % type: string
# % description: Resampling to be used for overviews
# % options: NEAREST,​AVERAGE,​BILINEAR​,​CUBIC​,​CUBICSPLINE​,​LANCZOS​,​MODE,​RMS
# % required: no
# %end

# %option
# % key: minimal_overview_size
# % type: integer
# % description: Minimal size of overview tiles (default=256)
# % answer: 256
# % required: no
# %end

# %option
# % key: temporal_tree
# % type: string
# % description: Strftime format to create temporal directory name or tree (e.g. "%Y/%m/%d")
# % required: no
# %end

# %option G_OPT_F_SEP
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: f
# % label: Force export of floating point maps as Float32
# % description: Force export of floating point maps as Float32
# %end

# %flag
# % key: s
# % label: Use semantic label in directory structure
# % description: Use semantic label in directory structure
# %end

# %flag
# % key: n
# % label: Export NoData maps as Byte
# % description: Export NoData maps as Byte, default is to skip export of maps with only NoData
# %end

import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs
import grass.temporal as tgis
from grass.pygrass.modules.interface import Module

OVERWRITE = False


def get_target_directory(
    map_row: dict,
    output_directory: Path | None = None,
    temporal_tree: str = "%Y/%m/%d",
    use_semantic_label: bool = False,
) -> Path:
    """Get target directory tree for raster map."""
    if use_semantic_label:
        output_directory /= map_row["semantic_label"]
    return output_directory / map_row["start_time"].strftime(temporal_tree)


def check_datatype(
    map_info: dict,
    force_float32: bool = False,
    export_empty_as_byte: bool = False,
) -> tuple[str, int | None]:
    """Check the datatype of the map and return the smallest appropriate GDAL type.

    :param map_info: Dictionary containing map information
    :param force_float32: Force the use of Float32 type for all floating point maps
    """
    int_ranges = {
        "Byte": (0, 255),
        "UInt16": (0, 65535),
        "Int16": (-32768, 32767),
        "UInt32": (0, 4294967295),
        "Int32": (-2147483648, 2147483647),
    }
    if not map_info["min"] or not map_info["max"]:
        return "Byte" if export_empty_as_byte else None, 255
    # Check for integer types
    if map_info["datatype"] == "CELL":
        # Check for integer types
        # Int8 is not yet supported by r.out.gdal
        for dtype, int_range in int_ranges.items():
            if map_info["min"] >= int_range[0] and map_info["max"] < int_range[1]:
                return dtype, int_range[1]
            if map_info["min"] > int_range[0] and map_info["max"] <= int_range[1]:
                return dtype, int_range[0]

    # Check for floating point types
    elif map_info["datatype"] == "FCELL" or force_float32:
        return "Float32", None
    return "Float64", None


def export_map_row(
    map_row: dict,
    output_directory: str = ".",
    flags: str = "cm",
    raster_format: str = "COG",
    compression: str = "LZW",
    blocksize: int = 256,  # must be multiple of 16
    resampling: str | None = None,
    level: int | None = None,
    separator: str = "|",
    temporal_tree: str = "%Y/%m/%d",
    overview_min_size: int = 256,
    truncate_float: bool = False,
    use_semantic_label: bool = False,
    export_empty_as_byte: bool = False,
) -> str:
    """Export raster map using r.out.gdal.

    The function applies the most appropriate create options
    based on the raster map type, range and GDAL format driver.
    See:
    https://gdal.org/en/stable/drivers/raster/cog.html
    https://gdal.org/en/stable/drivers/raster/gtiff.html
    """
    # Check the Raster map type and range
    raster_info = gs.raster_info(map_row["name"])
    data_type, no_data = check_datatype(
        raster_info,
        force_float32=truncate_float,
        export_empty_as_byte=export_empty_as_byte,
    )
    if not data_type:
        gs.warning(_("Map {} is empty. Skipping export...").format(map_row["name"]))
        return None
    target_directory = get_target_directory(
        map_row,
        Path(output_directory),
        temporal_tree=temporal_tree,
        use_semantic_label=use_semantic_label,
    )

    # Get number of overviews
    overview_list = []
    overview = 2
    while (
        float(raster_info["rows"]) / overview > overview_min_size
        and float(raster_info["rows"]) / overview > overview_min_size
    ):
        overview_list.append(overview)
        overview *= 2

    # Allways create BIGTIFFs
    createopt = "BIGTIFF=YES"
    # Set the resampling method
    if resampling:
        createopt += f",RESAMPLING={resampling}"
    # Default to NEAREST for Byte, AVERAGE for others
    elif data_type == "Byte":
        createopt += ",RESAMPLING=NEAREST"
    else:
        createopt += ",RESAMPLING=CUBIC"
    # Set the compression type and make sure a predictor is used
    createopt += f",COMPRESS={compression}"

    # Set GTiff specific create options
    if raster_format == "GTiff":
        createopt += ",TILED=YES"
        if compression in {"DEFLATE", "LERC_DEFLATE"} and level:
            createopt += f",ZLEVEL={level}"
        elif compression in {"ZSTD", "LERC_ZSTD"} and level:
            createopt += f",ZSTD_LEVEL={level}"
        if data_type in {"Float32", "Float64"}:
            createopt += ",PREDICTOR=3"
        else:
            createopt += ",PREDICTOR=2"
        createopt += f",BLOCKXSIZE={blocksize},BLOCKYSIZE={blocksize}"
    else:
        createopt += f",BLOCKSIZE={blocksize}"
        # Statistics are useful for registering the map in an STRDS
        if (
            compression in {"DEFLATE", "ZSTD", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"}
            and level
        ):
            createopt += f",LEVEL={level}"
        createopt += ",PREDICTOR=YES,STATISTICS=YES"

    if data_type == "Float32" and truncate_float and "f" not in flags:
        flags += "f"

    output_file = target_directory / f"{map_row['name']}.tif"
    if output_file.exists() and not OVERWRITE:
        gs.warning(
            _(
                "Output file {} already exists and overwrite is false, skipping...",
            ).format(str(output_file)),
        )
    else:
        export_module = Module(
            "r.out.gdal",
            flags=flags,
            input=map_row["name"],
            output=str(output_file),
            format=raster_format,
            nodata=no_data,
            type=data_type,
            overwrite=OVERWRITE,
            createopt=createopt,
            overviews=len(overview_list),
            quiet=True,
            stderr_=PIPE,
        )
        if export_module.returncode != 0:
            gs.fatal(
                _(
                    "Exporting map {} failed with error:\n{}",
                ).format(map_row["name"], export_module.stderr),
            )
    return separator.join(
        [
            map_row["name"],
            map_row["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            (
                map_row["end_time"].strftime("%Y-%m-%d %H:%M:%S")
                if map_row["end_time"]
                else ""
            ),
            map_row["semantic_label"] or "",
            str(output_file),
        ],
    )


def main() -> None:
    """Do the main work."""
    options, flags = gs.parser()
    global OVERWRITE
    OVERWRITE = gs.overwrite()
    nprocs = int(options["nprocs"])
    output_directory = Path(options["output_directory"])
    # Check if maps are exported to GDAL formats
    temporal_tree = options["temporal_tree"] or "%Y/%m/%d"

    tgis.init()
    input_strds = tgis.open_old_stds(options["input"], "strds")
    input_strds_maps = input_strds.get_registered_maps(
        columns="name,start_time,end_time,semantic_label",
        where=options["where"],
    )
    if not input_strds_maps:
        gs.warning(_("No maps found in the space-time raster dataset."))
        return

    # Create the output directory structure if it does not exist
    # Doing this sequentialy to avoid race condisions
    output_directories = {
        get_target_directory(
            map_row,
            output_directory,
            temporal_tree=temporal_tree,
            use_semantic_label=flags["s"],
        )
        for map_row in input_strds_maps
    }
    for output_dir in output_directories:
        output_dir.mkdir(parents=True, exist_ok=True)
    export_map_row_tif = partial(
        export_map_row,
        output_directory=str(output_directory),
        flags="cm",
        raster_format=options["format"],
        compression=options["compression"],
        level=int(options["level"]) if options["level"] else None,
        truncate_float=flags["f"],
        separator=gs.separator(options["separator"]),
        temporal_tree=options["temporal_tree"] or "%Y/%m/%d",
        overview_min_size=int(options["minimal_overview_size"]),
        use_semantic_label=flags["s"],
        export_empty_as_byte=flags["n"],
    )

    # Export maps to GeoTIFF / COG
    if nprocs > 1:
        with Pool(nprocs) as pool:
            register_strings = pool.map(
                export_map_row_tif,
                (dict(input_map) for input_map in input_strds_maps),
            )
    else:
        register_strings = [export_map_row_tif(map_row) for map_row in input_strds_maps]

    # Print register information
    print(
        "\n".join(
            [register_string for register_string in register_strings if register_string]
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
