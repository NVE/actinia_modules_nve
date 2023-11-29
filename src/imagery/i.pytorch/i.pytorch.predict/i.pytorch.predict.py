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
 - unclear parallelization / tiling

 time i.pytorch.predict device=cpu group=image_group output= tile_size= nprocs=
 https://github.com/NVE/Snotjeneste/blob/nve-cop/run_dl.py

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

# %option
# %key: tile_size
# % type: integer
# % required: yes
# % multiple: yes
# % description: Number of cows and columns in tiles
# % answer: 512,512
# %end

# %option
# %key: device
# % type: string
# % required: yes
# % multiple: no
# % description: Device to use for prediction (CPU or GPU), currently only CPU is supported
# % options: CPU,GPU
# % answer: CPU
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


import json
import shutil
import sys

from datetime import datetime
from functools import partial
from pathlib import Path
from multiprocessing import Pool

import grass.script as gs
from grass.pygrass.raster import raster2numpy, numpy2raster
from grass.pygrass.gis.region import Region


TMP_NAME = gs.tempname(12)


# Read and check configuration
def read_config(
    json_path,
    input_group,
):
    """Read band configuration for input deep learning model
    Example for configuration:
    {"model_type": "UNET",
    "valid_output_range": [0,100],
    "bands":
        {"red": [1, {"offset": 0, "scale": 1, "valid_range": [0, 255]}],
        "blue": [2, {"offset": 0, "scale": 1, "valid_range": [0, 255]}],
        },
    "n_classes": 2,
    # "in_channels": 1,  # Could be derived from bands
    "out_channels": 1,  # Could be derived from bands
    "depth": 5,
    "start_filts": 32,
    "up_mode": "bilinear",
    "merge_mode": "concat",
    "partial_conv": True,
    "use_bn": True,
    "activation_func": "leaky_relu"
    }
    """
    if not json_path.exists():
        gs.fatal(_("Could not open "))
    config = json.loads(json_path.read_text())

    # Validate config file
    required_keys = [
        "valid_output_range",
        "bands",
        "n_classes",
        "depth",
        "start_filts",
        "up_mode",
        "merge_mode",
        "partial_conv",
        "use_bn",
        "activation_func",
    ]
    required_band_keys = ["valid_range", "offset", "scale"]
    for config_key in required_keys:
        if config_key not in config:
            gs.fatal(
                _("Key '{0}' missing in input config file {1}").format(
                    config_key, str(json_path)
                )
            )

    maps_in_group = (
        gs.read_command("i.group", group=input_group, flags="g", quiet=True)
        .strip()
        .split()
    )

    input_group_dict = {}
    semantic_labels = []
    for raster_map in maps_in_group:
        raster_map_info = gs.raster_info(raster_map)
        semantic_label = raster_map_info["semantic_label"]
        if semantic_label not in config["bands"]:
            continue
        semantic_labels.append(semantic_label)
        for band_key in required_band_keys:
            if band_key not in config["bands"][semantic_label][1]:
                gs.fatal(
                    _("Description of band <{0}> lacks key <{1}>").format(
                        semantic_label, band_key
                    )
                )
        valid_range = config["bands"][semantic_label][1]["valid_range"]
        if raster_map_info["min"] < valid_range[0]:
            gs.warning(
                _(
                    "Minimum of raster map <{0}> {1} exeeds lower bound ({2}) of valid range"
                ).format(raster_map, raster_map_info["min"], valid_range[0])
            )
        if raster_map_info["max"] > valid_range[1]:
            gs.warning(
                _(
                    "Maximum of raster map <{0}> {1} exeeds upper bound ({2}) of valid range"
                ).format(raster_map, raster_map_info["max"], valid_range[1])
            )
        input_group_dict[config["bands"][semantic_label][0]] = (
            raster_map,
            raster_map_info["datatype"],
            config["bands"][semantic_label][1],
        )

    for band in config["bands"]:
        if band not in semantic_labels:
            gs.fatal(
                _("Band {0} is missing in input group {1}").format(band, input_group)
            )
    return config, input_group_dict


def align_bbox_region(bbox, reference_region=None, overlap=0):
    """Align a boundig box to the current or a reference region"""
    if not reference_region:
        reference_region = gs.region()
    bbox["w"] = (
        np.floor((bbox["w"] - reference_region["w"]) / reference_region["ewres"])
        * reference_region["ewres"]
        + reference_region["w"]
        - overlap * reference_region["ewres"]
    )
    bbox["e"] = (
        np.ceil((bbox["e"] - reference_region["e"]) / reference_region["ewres"])
        * reference_region["ewres"]
        + reference_region["e"]
        + overlap * reference_region["ewres"]
    )
    bbox["s"] = (
        np.floor((bbox["s"] - reference_region["s"]) / reference_region["nsres"])
        * reference_region["nsres"]
        + reference_region["s"]
        - overlap * reference_region["nsres"]
    )
    bbox["n"] = (
        np.ceil((bbox["n"] - reference_region["n"]) / reference_region["nsres"])
        * reference_region["nsres"]
        + reference_region["n"]
        + overlap * reference_region["nsres"]
    )
    return bbox


def create_tiling(tile_rows, tile_cols, overlap=128, region=None):
    """Create tiling aligned to a given or current region with
    tiles of a fixed number of rows and column and at least
    overlap number of pixels around
    Returns a dictionary with inner and outer region of a tile
    """
    gs.verbose(_("Setting up tiling ..."))
    reg = region or gs.region()
    core_reg = gs.parse_command(
        "g.region", flags="ug", grow=-overlap, overwrite=True, quiet=True
    )
    tile_cols_n = np.ceil(float(core_reg["cols"]) / (tile_cols - 2.0 * overlap)).astype(
        int
    )
    tile_rows_n = np.ceil(float(core_reg["rows"]) / (tile_rows - 2.0 * overlap)).astype(
        int
    )

    env = os.environ.copy()
    env["GRASS_REGION"] = gs.region_env(grow=-overlap)

    gs.run_command(
        "v.mkgrid",
        quiet=True,
        grid=[tile_cols_n, tile_rows_n],
        map=TMP_NAME,
        overwrite=True,
        env=env,
    )
    tile_setup = np.genfromtxt(
        gs.read_command(
            "v.to.db", option="bbox", flags="p", map=TMP_NAME, overwrite=True
        )
        .strip("\n")
        .lower()
        .split("\n"),
        names=True,
        delimiter="|",
        dtype=None,
    )
    dtype_names = tile_setup.dtype.names
    tiling_dict = {}
    for tile_coords in tile_setup:
        inner_tile_bbox = align_bbox_region(
            {dtype_name: tile_coords[dtype_name] for dtype_name in dtype_names},
            overlap=0,
        )
        tile_bbox = align_bbox_region(
            {dtype_name: tile_coords[dtype_name] for dtype_name in dtype_names},
            overlap=128,
        )
        if tile_bbox["n"] >= reg["n"]:
            # Top row
            tile_bbox["n"] = reg["n"]
            tile_bbox["s"] = reg["n"] - (tile_rows * reg["nsres"])
            inner_tile_bbox["n"] = reg["n"]
        elif tile_bbox["s"] <= reg["s"]:
            # Bottom row
            tile_bbox["s"] = reg["s"]
            tile_bbox["n"] = reg["s"] + (tile_rows * reg["nsres"])
            inner_tile_bbox["s"] = reg["s"]
        else:
            missing_rows = tile_rows - (tile_bbox["n"] - tile_bbox["s"]) / reg["nsres"]
            if missing_rows % 2 > 0.0:
                add_rows = [np.floor(missing_rows / 2), np.floor(missing_rows / 2) + 1]
            else:
                add_rows = [missing_rows / 2] * 2
            tile_bbox["s"] = tile_bbox["s"] - (add_rows[0] * reg["nsres"])
            tile_bbox["n"] = tile_bbox["n"] + (add_rows[1] * reg["nsres"])
        tile_bbox["rows"] = int((tile_bbox["n"] - tile_bbox["s"]) / reg["nsres"])
        inner_tile_bbox["rows"] = int(
            (inner_tile_bbox["n"] - inner_tile_bbox["s"]) / reg["nsres"]
        )
        if tile_bbox["e"] >= reg["e"]:
            # Right column
            tile_bbox["e"] = reg["e"]
            tile_bbox["w"] = reg["e"] - (tile_rows * reg["ewres"])
            inner_tile_bbox["e"] = reg["e"]
        elif tile_bbox["w"] <= reg["w"]:
            # Left column
            tile_bbox["w"] = reg["w"]
            tile_bbox["e"] = reg["w"] + (tile_rows * reg["ewres"])
            inner_tile_bbox["w"] = reg["w"]
        else:
            missing_cols = tile_cols - (tile_bbox["e"] - tile_bbox["w"]) / reg["ewres"]
            if missing_cols % 2 > 0.0:
                add_cols = [np.floor(missing_cols / 2), np.floor(missing_cols / 2) + 1]
            else:
                add_cols = [missing_cols / 2] * 2
            tile_bbox["w"] = tile_bbox["w"] - (add_cols[0] * reg["ewres"])
            tile_bbox["e"] = tile_bbox["e"] + (add_cols[1] * reg["ewres"])
        tile_bbox["cols"] = int((tile_bbox["e"] - tile_bbox["w"]) / reg["ewres"])
        inner_tile_bbox["cols"] = int(
            (inner_tile_bbox["e"] - inner_tile_bbox["w"]) / reg["ewres"]
        )
        tiling_dict[tile_bbox["cat"]] = {}
        tiling_dict[tile_bbox["cat"]]["inner"] = inner_tile_bbox
        tiling_dict[tile_bbox["cat"]]["outer"] = tile_bbox
    return tiling_dict


def predict(np_cube, dl_model=None):
    """Dummy function for testing parallel processing
    Should be replaced by something like predict function in:
    https://github.com/NVE/Snotjeneste/blob/nve-cop/run_dl.py
    """
    print(dl_model)
    return np.sum(np_cube, axis=2)


def load_model(dl_model_path):
    """The following should be included in a predict function"""
    # load pytorch model
    if not dl_model_path.exists():
        gs.fatal(("Model file {} not found").format(str(dl_model_path)))
    dl_model = UNet(
        n_classes=dl_config["n_classes"],
        in_channels=len(dl_config["bands"]),
        depth=dl_config["depth"],
        use_bn=True,
        partial_conv=True,
    )
    dl_model.load_state_dict(
        torch.load(str(dl_model_path), map_location=lambda storage, loc: storage)
    )
    dl_model.to(torch.device(options["device"]))
    dl_model.eval()
    out = _tiled_prediction(
        data_cube,
        dl_model,
        [512, 512],
        [128, 128],
        apply_softmax=(not reg),
        apply_classifier=(not reg),
    ).squeeze()
    # Clip result to valid output range
    out = np.clip(out, *dl_config["valid_output_range"])


def read_bands(raster_map_dict, bbox):
    """Read band maps and return stacked numpy array for all bands
    after applying nan-replacement, scale, offset and clamping to valid range"""
    # pygrass sets region for pygrass tasks
    pygrass_region = Region()
    raster_region = deepcopy(pygrass_region)
    raster_region.set_bbox(Bbox(bbox["n"], bbox["s"], bbox["e"], bbox["w"]))
    raster_region.set_raster_region()
    # Clip to valid range
    data_cube = []
    for band_number in sorted(raster_map_dict.keys()):
        npa = raster2numpy(raster_map_dict[band_number][0])
        # Set null to nan
        if raster_map_dict[band_number][1] == "CELL":
            npa = np.where(npa == -2147483648, np.nan, npa)
        # Add offset ???
        if raster_map_dict[band_number][2]["offset"] != 0:
            npa = npa + raster_map_dict[band_number][3]
        # Apply scale ???
        if raster_map_dict[band_number][2]["scale"] != 1:
            npa = npa * raster_map_dict[band_number][4]
        # Clip to valid range ???
        if raster_map_dict[band_number][2]["valid_range"]:
            npa = np.clip(npa, *raster_map_dict[band_number][2]["valid_range"])
        data_cube.append(npa)
    return np.stack(data_cube, axis=-1)


def write_result(np_array, map_type, map_name, bbox):
    """"""
    gs.use_temp_region()
    gs.run_command(
        "g.region",
        n=bbox["n"],
        w=bbox["w"],
        e=bbox["e"],
        s=bbox["s"],
        nsres=bbox["nsres"],
        ewres=bbox["ewres"],
        flags="g",
    )
    numpy2raster(np_array, map_type, map_name, overwrite=True)
    gs.del_temp_region()
    return 0


def tiled_rediction(bbox, raster_maps=None):
    """Predict function to be parallelized"""
    data_cube = read_bands(raster_maps, bbox)
    write_result(predict(data_cube), "CELL", f"{TMP_NAME}_{bbox['cat']}", bbox)
    return 0


def patch_results(output_map):
    """Patch resulting raster maps"""
    input_maps = (
        gs.read_command("g.list", type="raster", pattern=f"{TMP_NAME}*")
        .strip()
        .split("\n")
    )
    gs.run_command(
        "r.patch", input=input_maps, output=output_map, verbose=True, nprocs=8
    )
    gs.raster_history(output_map, overwrite=True)


def main():
    """Do the main work"""

    # Check device
    if options["device"] != "CPU":
        gs.fatal(_("Currently only CPU device is supported"))

    # Get comutational region
    region = gs.parse_command("g.region", flags="ug")

    # Parse and check configuration
    dl_config, group_dict = read_config(options["band_configuration"], options["group"])

    # Check if mask is active
    # ???

    # Check tile size
    if options["tile_size"]:
        try:
            tile_size = list(map(int, options["tile_size"].split(",")))
            # Create tiling
            tile_set = create_tiling(tile_size, overlap=128, region=None)
        except ValueError:
            gs.fatal(_("Invalid input in tile_size option"))
    else:
        tile_set = create_tiling(tile_size, overlap=128, region=None)

    raster_maps = list(dl_config["bands"].keys())
    raster_maps = ["test_b1", "test_b2", "test_b3"]

    tiled_group_rediction = partial(tiled_rediction, raster_maps=raster_maps)
    inner_tiles = {cat: tile_set[cat]["inner"] for cat in tile_set}
    nprocs = np.min(int(options["nprocs"]), len(tile_set))
    if nprocs > 1:
        with Pool(nprocs) as pool:
            pool.map(tiled_group_rediction, inner_tiles.values())
        patch_results(options["output"])
    else:
        [tiled_group_rediction(tile) for til in inner_tiles.values()]
        gs.run_command("g.rename", raster=f"{TMP_NAME}, {options['output']}")


if __name__ == "__main__":
    options, flags = gs.parser()
    # Lazy imports
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.autograd import Variable
    except ImportError:
        gs.fatal(("Could not import pytorch. Please make sure it is installed."))
    import numpy as np

    try:
        from unet import UNet
    except ImportError:
        gs.fatal(
            (
                "Could not import included unet library. Please check the addon installation."
            )
        )

    sys.exit(main())
