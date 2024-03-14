#!/usr/bin/env python3

"""
 MODULE:       i.pytorch.predict
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Apply Deep Learning model to imagery group using pytorch
 COPYRIGHT:    (C) 2023 by Stefan Blumentrath

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

 ToDo:
 - linting
 - test case

"""

# %module
# % description: Apply Deep Learning model to imagery group using pytorch
# % keyword: raster
# % keyword: imagery
# % keyword: deep learning
# % keyword: pytorch
# % keyword: unet
# % keyword: GPU
# % keyword: predict
# %end

# %option G_OPT_I_GROUP
# %key: input
# % description: Input imagery group
# %end

# %option G_OPT_F_INPUT
# %key: model
# % type: string
# % required: yes
# % multiple: no
# % description: Path to input deep learning model file (.pt)
# %end

# %option G_OPT_F_INPUT
# %key: model_code
# % type: string
# % required: yes
# % multiple: no
# % description: Path to input deep learning model code (.py)
# %end

# %option
# %key: tile_size
# % type: integer
# % required: no
# % multiple: yes
# % description: Number of rows and columns in tiles
# %end

# %option
# %key: overlap
# % type: integer
# % required: no
# % multiple: no
# % description: Number of rows and columns of overlap in tiles
# %end

# %option G_OPT_F_INPUT
# %key: mask_json
# % required: no
# % multiple: no
# % description: JSON file with one or more mask band or map name(s) and reclass rules for masking, e.g. {"mask_band": "1 thru 12 36 = 1", "mask_map": "0"}
# %end

# %option G_OPT_F_INPUT
# %key: configuration
# % type: string
# % required: yes
# % multiple: no
# % description: Path to JSON file with band configuration in the input deep learning model
# %end

# %option G_OPT_M_NPROCS
# %end

# %option G_OPT_R_OUTPUT
# %end

# %flag
# %key: c
# % description: Use CPU as device for prediction, default is use cuda (GPU) if detected
# %end

# %flag
# %key: l
# % description: Limit output to valid range (data outside the valid range is set to valid min/max)
# %end

# To do:
# optimize tiling (shape) to minimize number of tiles and overlap between them within max size
# Handle divisible by x (e.g. 8) for tile size to avoid:
#    RuntimeError: Sizes of tensors must match except in dimension 1


import atexit
import json
import sys

from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path

# from multiprocessing import Pool

import grass.script as gs
from grass.pygrass.raster import raster2numpy, numpy2raster
from grass.pygrass.gis.region import Region
from grass.pygrass.vector import Bbox


TMP_NAME = gs.tempname(12)


# Read and check configuration
def read_config(module_options):
    """Read band configuration for input deep learning model
    Example for configuration se manual:
    {"model":  # Dictinary matching the parameters of the signature of the UNet model code
       {
        "type": "UNetV1",  # Name of the class representing the UNet model in the model code
        # "in_channels": 1,  # Number of input chanels can be derived from 'input_bands' section
        # "out_channels": 1,  # Number of output chanels can be derived from 'output_bands' section
        # Keys below depend on the UNet model code, which should consequently use keyword arguments with defined data type and defaults for all parameters
        "n_classes": 2,
        "depth": 5,
        "start_filts": 32,
        "up_mode": "bilinear",
        "merge_mode": "concat",
        "partial_conv": True,
        "use_bn": True,
        "activation_func": "lrelu"
        },
    "input_bands": {  # dictionary describing the input bands expected by the model
        'S1_reflectance_an':  {  # input band name / key
            "order": 1,  # position in the input data cube to the DL model
            "offset": 0,  # Offset to be applied to the values in the input band, (0 means no offset)
            "scale": 1,  # Scale to be applied to the values in the input band, after offset (1 means no scaling)
            "valid_range": [None, 2],  # Tuple with valid range (min, max) of the input data with scale and offset applied, (None means inf for max and -inf for min)
            "fill_value": 0,  # Value to replace NoData value with
            "description": "Sentinel-3 SLSTR band S1 scaled to reflectance values",  # Human readable description of the band that is expected as input
            },
        'S2_reflectance_an':  {"order": 2, "offset": 0, "scale": 1, "valid_range": [None, 2], "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        'S3_reflectance_an':  {"order": 3, "offset": 0, "scale": 1, "valid_range": [None, 2], "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        'S5_reflectance_an':  {"order": 4, "offset": 0, "scale": 1, "valid_range": [None, 2], "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        'S6_reflectance_an':  {"order": 5, "offset": 0, "scale": 1, "valid_range": [None, 2], "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        'S8_BT_in':  {"order": 6, "offset": -200, "scale": 100, "valid_range": None, "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        'S9_BT_in':  {"order": 7, "offset": -200, "scale": 100, "valid_range": None, "description": "Sentine-3 SLSTR band S1 in reflectance values"},
        },
    "output_bands":  # This section contains meta information about output bands from the DL model, each output band should have a description
        {
            "fsc": {
                "title": "Fractional Snow Cover (FSC)",
                "standard_name": "surface_snow_area_fraction",  # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
                "model_name": "NVECOP2-CNN",
                "model_version": "v1.0",
                "keywords": ["fractional snow cover", "earth observation", "Sentinel-3", "SLSTR"],
                "description": ("The NVECOP2-CNN FSC algorithm is developed by NR in
                    "the AI4Arctic project based on a CNN."
                    "A modified version of the UNet has been applied and trained with"
                    "selected and quality-controlled 10-m resolution snow maps based"
                    "on Sentinel-2."
                    "The FSC product provides regular information on snow "
                    "cover fraction (0-100 %) per grid cell for the given "
                    "land area except for land ice areas. The product is "
                    "based on reflectance values from Sentinel-3 SLSTR RBT product,"
                    "bands 1,2,3,5, and 6 at 500m resolution"),
                "units": "percent",  # ideally CF-compliant name or symbol: https://ncics.org/portfolio/other-resources/udunits2/
                "valid_output_range": [0,100],
                "dtype": "uint8",  # string representing the numpy dtype ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64")
                "fill_value": 255  # Value representing NoData
                "offset": 0,  # Offset to be applied to the values in the output band, (0 means no offset)
                "scale": 1,  # Scale to be applied to the values in the output band, after offset (1 means no scaling)
                "classes": None,  # class values and names if output band contains classes, e.g. {0: "bare land", 1: "snow cover", 2: "waterbody", 3: "cloud"}
                },
        },
    }
    """
    json_path = Path(module_options["configuration"])
    input_group = module_options["input"]
    if not json_path.exists():
        gs.fatal(_("Could not find configuration file <{}>").format(str(json_path)))

    # Validate config file
    config, backbone, model_kwargs = validate_config(
        json_path, Path(module_options["model_code"])
    )

    maps_in_group = (
        gs.read_command("i.group", group=input_group, flags="g", quiet=True)
        .strip()
        .split()
    )

    input_group_dict = {}
    semantic_labels = []
    masks = None
    mask_rules = None
    if module_options["mask_json"]:
        mask_rules = {}
        with open(module_options["mask_json"]) as mask_json:
            masks = json.load(mask_json)

    for raster_map in maps_in_group:
        raster_map_info = gs.raster_info(raster_map)
        semantic_label = raster_map_info["semantic_label"]
        if mask_rules is not None and masks and semantic_label in masks:
            mask_rules[semantic_label] = {raster_map: masks[semantic_label]}
        if semantic_label not in config["input_bands"]:
            continue
        semantic_labels.append(semantic_label)
        valid_range = config["input_bands"][semantic_label]["valid_range"]
        if valid_range and valid_range[0] and raster_map_info["min"] < valid_range[0]:
            gs.warning(
                _(
                    "Minimum of raster map <{0}> {1} exeeds lower bound ({2}) of valid range"
                ).format(raster_map, raster_map_info["min"], valid_range[0])
            )
        if valid_range and valid_range[1] and raster_map_info["max"] > valid_range[1]:
            gs.warning(
                _(
                    "Maximum of raster map <{0}> {1} exeeds upper bound ({2}) of valid range"
                ).format(raster_map, raster_map_info["max"], valid_range[1])
            )
        input_group_dict[config["input_bands"][semantic_label]["order"]] = (
            raster_map,
            raster_map_info["datatype"],
            config["input_bands"][semantic_label],
        )

    for band in config["input_bands"]:
        if band not in semantic_labels:
            gs.fatal(
                _("Band {0} is missing in input group {1}").format(band, input_group)
            )

    if masks:
        for mask_entry in masks:
            if gs.find_file(mask_entry)["fullname"]:
                mask_rules[mask_entry] = {mask_entry: masks[mask_entry]}
        # Check if all required masks are found:
        for mask_entry in masks:
            if mask_entry not in mask_rules:
                gs.fatal(
                    _("No raster map found for requested mask {}").format(mask_entry)
                )
        mask_rules_list = []
        for idx, mask_config in enumerate(mask_rules.values()):
            raster_map, rules = list(mask_config.items())[0]
            if any([symbol in rules for symbol in "<>=,"]):
                mask_rules_list.extend(
                    [f"{raster_map}{rule}" for rule in rules.split(",")]
                )
            else:
                reclass_map = f"{TMP_NAME}_mask_{idx}"
                gs.write_command(
                    "r.reclass",
                    input=raster_map,
                    output=reclass_map,
                    rules="-",
                    stdin=f"{rules} = 1\n",
                )
                mask_rules_list.append(reclass_map)
        mask_rules = " && ".join(mask_rules_list)

    return config, backbone, model_kwargs, input_group_dict, mask_rules


def apply_mask(input_map, output_map, fill_value, masking):
    """Apply mask(s) to the input map, and replace fill_value
    to produce the output map"""
    masked_pixels = f"if({masking}, {input_map}, null())" if masking else input_map
    valid_pixels = (
        f"if({input_map}=={fill_value}, null(),{masked_pixels})"
        if fill_value
        else input_map
    )
    gs.mapcalc(f"{output_map}={valid_pixels}")


def create_tiling(tile_rows, tile_cols, overlap=128, region=None):
    """Create tiling aligned to a given or current region with
    tiles of a fixed number of rows and column and at least
    overlap number of pixels around
    Returns a dictionary with inner and outer region of a tile
    """
    reg = region or gs.region()

    tiling_dict = {}

    overlap_distance_x = overlap * float(reg["ewres"])
    overlap_distance_y = overlap * float(reg["nsres"])

    inner_tile_cols = tile_cols - 2.0 * overlap
    inner_tile_rows = tile_rows - 2.0 * overlap

    inner_tile_extent_x = inner_tile_cols * float(reg["ewres"])
    inner_tile_extent_y = inner_tile_rows * float(reg["nsres"])

    tile_cols_n = float(reg["cols"]) / inner_tile_cols
    tile_rows_n = float(reg["rows"]) / inner_tile_rows

    gs.verbose(
        _("Setting up tiling {rows} rows and {columns} columns ...").format(
            rows=np.ceil(tile_rows_n).astype(int),
            columns=np.ceil(tile_cols_n).astype(int),
        )
    )

    x_offset = ((np.ceil(tile_cols_n) - tile_cols_n) * (inner_tile_cols)) / 2.0
    y_offset = ((np.ceil(tile_rows_n) - tile_rows_n) * (inner_tile_rows)) / 2.0

    for cat, tile_idx in enumerate(
        list(
            product(
                range(np.ceil(tile_cols_n).astype(int)),
                range(np.ceil(tile_rows_n).astype(int)),
            )
        )
    ):
        tiling_dict[cat + 1] = {
            "inner": {
                "cat": cat + 1,
                "n": float(reg["n"])
                + np.ceil(y_offset) * float(reg["nsres"])
                - tile_idx[1] * inner_tile_extent_y,
                "s": float(reg["n"])
                + np.ceil(y_offset) * float(reg["nsres"])
                - (tile_idx[1] + 1) * inner_tile_extent_y,
                "w": float(reg["w"])
                - np.ceil(x_offset) * float(reg["ewres"])
                + tile_idx[0] * inner_tile_extent_x,
                "e": float(reg["w"])
                - np.ceil(x_offset) * float(reg["ewres"])
                + (tile_idx[0] + 1) * inner_tile_extent_x,
                "rows": int(inner_tile_rows),
                "cols": int(inner_tile_cols),
                "ewres": reg["ewres"],
                "nsres": reg["nsres"],
            },
            "outer": {
                "n": float(reg["n"])
                + np.ceil(y_offset) * float(reg["nsres"])
                + overlap_distance_y
                - tile_idx[1] * inner_tile_extent_y,
                "s": float(reg["n"])
                + np.ceil(y_offset) * float(reg["nsres"])
                - overlap_distance_y
                - (tile_idx[1] + 1) * inner_tile_extent_y,
                "w": float(reg["w"])
                - np.ceil(x_offset) * float(reg["ewres"])
                - overlap_distance_x
                + tile_idx[0] * inner_tile_extent_x,
                "e": float(reg["w"])
                - np.ceil(x_offset) * float(reg["ewres"])
                + overlap_distance_x
                + (tile_idx[0] + 1) * inner_tile_extent_x,
                "rows": int(inner_tile_rows + 2 * overlap),
                "cols": int(inner_tile_cols + 2 * overlap),
                "ewres": reg["ewres"],
                "nsres": reg["nsres"],
            },
        }
    return tiling_dict


def read_bands(raster_map_dict, bbox, null_value=0):
    """Read band maps and return stacked numpy array for all bands
    after applying nan-replacement, scale, offset and clamping to valid range"""
    # pygrass sets region for pygrass tasks
    pygrass_region = Region()
    raster_region = deepcopy(pygrass_region)
    # raster_region.read(force_read=True)
    raster_region.set_bbox(Bbox(bbox["n"], bbox["s"], bbox["e"], bbox["w"]))
    raster_region.set_raster_region()
    # Clip to valid range
    data_cube = []
    mask = None
    for band_number in sorted(raster_map_dict.keys()):
        npa = raster2numpy(raster_map_dict[band_number][0])
        # Set null to nan
        if raster_map_dict[band_number][1] == "CELL":
            npa = np.where(npa == -2147483648, np.nan, npa)
        if mask is None:
            mask = np.where(np.isnan(npa), True, False)
        else:
            mask = np.where(np.logical_or(mask, np.isnan(npa)), True, False)
        # Clip to valid range
        if raster_map_dict[band_number][2]["valid_range"]:
            min_value = raster_map_dict[band_number][2]["valid_range"][0]
            if not min_value:
                min_value = -np.inf
            max_value = raster_map_dict[band_number][2]["valid_range"][1]
            if not max_value:
                max_value = np.inf
            npa = np.clip(npa, min_value, max_value)

        # Abort if (any) map only contains nan
        if np.nansum(npa) == 0:
            return None, None

        # Add offset
        if raster_map_dict[band_number][2]["offset"] != 0:
            npa = npa + np.array(raster_map_dict[band_number][2]["offset"])

        # Apply scale
        if raster_map_dict[band_number][2]["scale"] != 1:
            npa = npa * np.array(raster_map_dict[band_number][2]["scale"])

        data_cube.append(npa)
    data_cube = np.stack(data_cube, axis=-1)
    data_cube[np.isnan(data_cube)] = null_value

    return data_cube, mask


def write_result(np_array, map_name, bbox):
    """Write prediction results to raster"""
    dtype2grass = {
        "uint8": "CELL",
        "int8": "CELL",
        "uint16": "CELL",
        "int16": "CELL",
        "uint32": "CELL",
        "int32": "CELL",
        "uint64": "CELL",
        "int64": "CELL",
        "float32": "FCELL",
        "float64": "DCELL",
    }
    pygrass_region = Region()
    raster_region = deepcopy(pygrass_region)
    raster_region.set_bbox(Bbox(bbox["n"], bbox["s"], bbox["e"], bbox["w"]))
    raster_region.set_raster_region()

    gs.use_temp_region()
    gs.run_command(
        "g.region",
        n=bbox["n"],
        w=bbox["w"],
        e=bbox["e"],
        s=bbox["s"],
        nsres=bbox["nsres"],
        ewres=bbox["ewres"],
    )

    numpy2raster(np_array, dtype2grass[np_array.dtype.name], map_name, overwrite=True)
    gs.del_temp_region()
    return 0


def tiled_prediction(
    idx,
    bboxes,
    group_dict=None,
    dl_config=None,
    dl_model=None,
    device=None,
    limit=False,
):
    """Predict function to be parallelized"""
    data_cube, mask = read_bands(group_dict, bboxes["outer"], null_value=0)
    if data_cube is None:
        return None
    prediction_result = predict_torch(
        data_cube, config_dict=dl_config, device=device, dl_model=dl_model
    )
    inner_mask = get_inner_bbox(mask, bboxes["outer"], bboxes["inner"])
    prediction_result = get_inner_bbox(
        prediction_result, bboxes["outer"], bboxes["inner"]
    )

    # Write each output band
    for idx, output_band in enumerate(dl_config["output_bands"]):
        output_dtype = dl_config["output_bands"][output_band]["dtype"]
        # Clip result to valid output range
        if dl_config["output_bands"][output_band]["valid_output_range"]:
            if limit:
                out_numpy = prediction_result[..., idx]
                # Limit to valid min
                out_numpy[
                    out_numpy
                    < dl_config["output_bands"][output_band]["valid_output_range"][0]
                ] = dl_config["output_bands"][output_band]["valid_output_range"][0]
                # Limit to valid max
                out_numpy[
                    out_numpy
                    > dl_config["output_bands"][output_band]["valid_output_range"][1]
                    & out_numpy
                    < 255
                ] = dl_config["output_bands"][output_band]["valid_output_range"][1]
            else:
                out_numpy = np.clip(
                    prediction_result[..., idx],
                    *dl_config["output_bands"][output_band]["valid_output_range"],
                )
        else:
            out_numpy = prediction_result[..., idx]
        if not output_dtype.startswith("float"):
            out_numpy[inner_mask] = dl_config["output_bands"][output_band]["fill_value"]
            out_numpy[np.isnan(out_numpy)] = dl_config["output_bands"][output_band][
                "fill_value"
            ]
            # out_numpy[np.isnan(out_numpy)] = dl_config["output_bands"][output_band]["fill_value"]
        else:
            out_numpy[inner_mask] = np.nan

        if out_numpy.dtype != np.dtype(output_dtype):
            if output_dtype.startswith("float"):
                out_numpy = np.round(
                    out_numpy, np.finfo(output_dtype).precision
                ).astype(output_dtype)
            else:
                out_numpy = np.round(out_numpy).astype(output_dtype)
        # Write data for inner tile
        write_result(
            out_numpy,
            f"{TMP_NAME}_{output_band}_{bboxes['inner']['cat']}",
            bboxes["inner"],
        )
    return 0


def patch_results(
    output_map, output_band, masking=None, fill_value=None, dl_config=None, nprocs=1
):
    """Patch resulting raster maps"""
    output_band_config = dl_config["output_bands"][output_band]
    output_map_name = f"{output_map}_{output_band}"
    patch_map_name = output_map_name if not masking else f"{TMP_NAME}_{output_map_name}"
    input_maps = (
        gs.read_command("g.list", type="raster", pattern=f"{TMP_NAME}_{output_band}*")
        .strip()
        .split("\n")
    )
    if len(input_maps) == 1:
        gs.run_command(
            "g.rename",
            raster=f"{input_maps[0]},{output_map_name}",
            quiet=True,
        )
    else:
        gs.run_command(
            "r.patch",
            input=input_maps,
            output=patch_map_name,
            overwrite=gs.overwrite(),
            quiet=True,
            nprocs=nprocs,
        )
    if masking:
        apply_mask(patch_map_name, output_map_name, fill_value, masking)

    if dl_config["output_bands"][output_band]["classes"]:
        gs.write_command(
            "r.category",
            map=output_map_name,
            rules="-",
            stdin="\n".join(
                f"{key}:{val}"
                for key, val in dl_config["output_bands"][output_band][
                    "classes"
                ].items()
            ),
            separator=":",
        )

    gs.raster_history(output_map_name, overwrite=True)
    gs.run_command(
        "r.support",
        map=output_map_name,
        title=output_band_config["title"],
        units=output_band_config["units"],
        source1=", ".join(
            [dl_config["model"]["model_name"], dl_config["model"]["model_version"]]
        ),
        description=output_band_config["description"],
        semantic_label=output_band_config["semantic_label"],
        overwrite=True,
        quiet=True,
    )


def get_inner_bbox(data_cube, outer_bbox, inner_bbox):
    """Get offset indices based on inner and outer (with overlap) bbox
    starting from upper left corner (n / w) and subset input data_cube
    using those indices"""
    offset_n = int((outer_bbox["n"] - inner_bbox["n"]) / outer_bbox["nsres"])
    offset_w = int((inner_bbox["w"] - outer_bbox["w"]) / outer_bbox["ewres"])
    bound_s = offset_n + inner_bbox["rows"]
    bound_e = offset_w + inner_bbox["cols"]

    if data_cube.ndim == 3:
        return data_cube[offset_n:bound_s, offset_w:bound_e, :]
    return data_cube[offset_n:bound_s, offset_w:bound_e]


def main():
    """Do the main work"""

    # Check device
    device = "cuda" if torch.cuda.is_available() and not flags["c"] else "cpu"

    # Get comutational region
    region = gs.parse_command("g.region", flags="ug")

    # Parse and check configuration
    dl_config, dl_backbone, dl_model_kwargs, group_dict, masking = read_config(options)

    # Load model
    dl_model = load_model(
        Path(options["model"]),
        dl_backbone,
        dl_model_kwargs,
        device="cpu" if flags["c"] else "gpu",
    )

    overlap = int(options["overlap"]) or 0

    # Check if mask is active
    # ???

    # Check tile size
    if options["tile_size"]:
        try:
            tile_size = list(map(int, options["tile_size"].split(",")))
            # Create tiling
            tile_set = create_tiling(*tile_size, overlap=overlap, region=None)
        except ValueError:
            gs.fatal(_("Invalid input in tile_size option"))
    else:
        tile_set = create_tiling(tile_size, overlap=overlap, region=None)

    tiled_group_rediction = partial(
        tiled_prediction,
        group_dict=group_dict,
        dl_config=dl_config,
        dl_model=dl_model,
        device=device,
        limit=flags["l"],
    )
    nprocs = np.min([int(options["nprocs"]), len(tile_set)])

    if device == "cpu":
        torch.set_num_threads(int(options["nprocs"]))
    # if nprocs > 1:
    #    with Pool(nprocs) as pool:
    #        pool.starmap(tiled_group_rediction, tile_set.items())
    # else:
    idx = 0
    for idx, tile_def in tile_set.items():
        gs.percent(idx, len(tile_set), 3)
        tiled_group_rediction(idx, tile_def)
        idx += 1

    # Patch or rename results and write metadata
    for output_band in dl_config["output_bands"]:
        patch_results(
            options["output"],
            output_band,
            masking=masking,
            fill_value=dl_config["output_bands"][output_band]["fill_value"],
            dl_config=dl_config,
            nprocs=nprocs,
        )


def cleanup():
    """Remove all temporary data"""
    # Remove Raster map files
    gs.run_command(
        "g.remove",
        type=["raster"],
        pattern=f"{TMP_NAME}*",
        flags="f",
        quiet=True,
    )
    # Remove external data if mapset uses r.external.out
    external = gs.parse_key_val(gs.read_command("r.external.out", flags="p"), sep=": ")
    if "directory" in external:
        for map_file in Path(external["directory"]).glob(
            f"{TMP_NAME}_*{external['extension']}"
        ):
            if map_file.is_file():
                map_file.unlink()


if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    # Lazy imports
    try:
        import torch
    except ImportError:
        gs.fatal(("Could not import pytorch. Please make sure it is installed."))
    try:
        import numpy as np
    except ImportError:
        gs.fatal(("Could not import pytorch. Please make sure it is installed."))

    gs.utils.set_path(modulename="i.pytorch", dirname="", path="..")
    try:
        from pytorchlib.utils import (
            # numpy2torch,
            # torch2numpy,
            validate_config,
            # transform_axes,
            load_model,
            predict_torch,
        )
    except ImportError:
        gs.fatal(
            (
                "Could not import included unet library. Please check the addon installation."
            )
        )

    sys.exit(main())
