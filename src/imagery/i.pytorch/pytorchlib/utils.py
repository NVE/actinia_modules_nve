#!/usr/bin/env python3

"""
MODULE:       pytorchlib.utils
AUTHOR(S):    Stefan Blumentrath
PURPOSE:      Collection of utility functions when working with torch
COPYRIGHT:    (C) 2023 by Stefan Blumentrath

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

"""

import inspect
import json
import sys
from copy import deepcopy
from importlib import import_module

import grass.script as gs

try:
    import torch
    import torch.nn.functional as F

    # from torch import nn
    # from torch.autograd import Variable
except ImportError:
    gs.fatal(_("Could not import pytorch. Please make sure it is installed."))
import numpy as np


def transform_axes(np_array, from_format="HWC", to_format="NCHW"):
    """Transform axes / dimensions of a numpy array
    between two formats described by single capital letters
    representing the order of axes / dimensions:
    N: Number of images in batch
    H: Height of the images
    W: Width of the images
    C: Number of channels / bands of the images

    Default is HWC to NCHW format
    """
    for format_string in [from_format, to_format]:
        if len(format_string) != len(set(format_string)):
            gs.fatal(_("Repeated axis in requested format"))
        if not set(format_string).issubset(set("NHWC")):
            gs.fatal(_("Only letters NHCW are allowed to describe dimension order"))
        if "H" not in format_string:
            gs.fatal(_("Height of images (H) is mandatory part of all formats"))
        if "W" not in format_string:
            gs.fatal(_("Width of images (W) is mandatory part of all formats"))
    if len(np_array.shape) != len(from_format):
        gs.fatal(_("Shape of input data and given format do not match"))
    reorder = {}
    reduce = []
    for letter in "NCHW":
        if letter in from_format:
            if letter not in to_format:
                # Reduce dimensions
                tmp_idx = len(to_format) + len(reduce)
                reorder[from_format.index(letter)] = tmp_idx
                reduce.append(tmp_idx)
            else:
                reorder[from_format.index(letter)] = to_format.index(letter)
        else:
            # Expand dimension (adding at the end)
            reorder[len(from_format)] = to_format.index(letter)
            np_array = np.expand_dims(np_array, len(from_format))
            from_format += letter
    # Reorder
    np_array = np.transpose(np_array, tuple(sorted(reorder, key=reorder.get)))
    if reduce:
        # Reduce dimensions if needed
        idx = []
        for dim in range(len(from_format)):
            if dim in reduce:
                idx.append(0)
                if np_array.shape[dim] > 1:
                    gs.warning(
                        _("Requested reduction of dimensions leads to loss of data")
                    )
            else:
                idx.append(slice(None))
        return np_array[tuple(idx)]
    return np_array


def numpy2torch(np_array, device="cpu", precision="float"):
    """Create torch variable from numpy array"""
    if precision not in ["float", "half"]:
        gs.fatal(
            _("Invalid precision {}. Only 'float' and 'half' are supported.").format(
                precision
            )
        )
    if precision == "float":
        return torch.from_numpy(np_array).float().to(device)
    return torch.from_numpy(np_array).half().to(device)


def torch2numpy(torch_tensor):
    """Create numpy array from torch variable"""
    if torch_tensor.device != "cpu":
        torch_tensor_numpy = deepcopy(torch.Tensor.cpu(torch_tensor).numpy())
    else:
        torch_tensor_numpy = deepcopy(torch_tensor.numpy())
    del torch_tensor
    return torch_tensor_numpy


def load_model_code(package_dir, object_name):
    """Load model class object using package directory and
    model object string as input"""
    if not package_dir.exists():
        gs.fatal(_("Package directory for module code not found"))
    sys.path.append(str(package_dir))
    module_name, class_name = object_name.rsplit(".", maxsplit=1)
    try:
        return getattr(import_module(module_name), class_name)
    except ImportError:
        gs.fatal(
            _("Could not import {object} from directory {dir}").format(
                object=object_name, dir=package_dir
            )
        )


def load_model(dl_model_path, dl_backbone, dl_kwargs, device="gpu"):
    """The following should be included in a predict function"""
    # load pytorch model
    if not dl_model_path.exists():
        gs.fatal(_("Model file {} not found").format(str(dl_model_path)))

    try:
        dl_model = dl_backbone(**dl_kwargs)
    except ValueError:
        gs.fatal(_("Configuration does not match model backbone code"))

    try:
        if device != "cpu":
            dl_model.load_state_dict(
                torch.load(
                    str(dl_model_path),
                )
            )
        else:
            dl_model.load_state_dict(
                torch.load(
                    str(dl_model_path),
                    map_location=torch.device("cpu"),
                )
            )
    except ValueError:
        gs.fatal(_("Model backbone and model file do not match"))

    return dl_model


def predict_torch(data_cube, config_dict=None, device=None, dl_model=None):
    """Apply a deep learning model to a numpy array on a given device (cpu or cuda)
    after applying the given transform function"""
    dl_model.to(device)
    dl_model.eval()
    if config_dict["model"]["model_dimensions"]["input_dimensions"] != "HWC":
        data_cube = transform_axes(
            data_cube,
            "HWC",
            config_dict["model"]["model_dimensions"]["input_dimensions"],
        )

    with torch.no_grad():
        data = numpy2torch(data_cube).to(device)
        torch_out = dl_model(data)
        del data
        # Apply extra ouptut transformations
        if "extra_ouptut_transformations" in config_dict["model"]:
            if "apply_softmax" in config_dict["model"]["extra_ouptut_transformations"]:
                torch_out = F.softmax(torch_out, dim=1)
            if (
                "apply_classifier"
                in config_dict["model"]["extra_ouptut_transformations"]
            ):
                _, torch_out = torch.max(torch_out, dim=1, keepdims=True)

    if config_dict["model"]["model_dimensions"]["output_dimensions"] != "HWC":
        return transform_axes(
            torch2numpy(torch_out),
            config_dict["model"]["model_dimensions"]["output_dimensions"],
            "HWC",
        )
    return torch2numpy(torch_out)


def not_in_types(data_type, allowed_types):
    """Check if data_type is not an element of allowed_types"""

    allowed_types = (
        (allowed_types,) if isinstance(allowed_types, type) else allowed_types
    )
    if data_type is None:
        data_type_str = "'None'"
    else:
        data_type_str = f"'{data_type.__class__.__name__}'"
    allowed_types_str = [
        f"'{data_type.__name__}'" if data_type else "'None'"
        for data_type in allowed_types
    ]
    if data_type_str not in allowed_types_str:
        return (data_type_str, allowed_types_str)
    return None


def validate_config(json_path, package_dir):
    """Check if configuration content matches an implemented
    torch backbone"""
    try:
        config_dict = json.loads(json_path.read_text())
    except RuntimeError:
        gs.fatal(
            _("Could not parse JSON content in configuration file <{}>").format(
                str(json_path)
            )
        )

    # Check model description
    if "model" not in config_dict:
        gs.fatal(_("'model' section missing in configuration"))
    for config_sub_key in [
        "model_backbone",
        "model_name",
        "model_dimensions",
        "model_version",
    ]:
        if config_sub_key not in config_dict["model"]:
            gs.fatal(
                _(
                    "Key '{0}' missing in 'model' section of configuration file <{1}>"
                ).format(config_sub_key, str(json_path))
            )
        if config_sub_key == "model_dimensions":
            for dim_description in ["input_dimensions", "output_dimensions"]:
                if dim_description not in config_dict["model"]["model_dimensions"]:
                    gs.fatal(
                        _(
                            "Key '{0}' missing in description of 'model_dimensions' in 'model' section of configuration file <{1}>"
                        ).format(dim_description, str(json_path))
                    )

    # Try to load model
    backbone = load_model_code(package_dir, config_dict["model"]["model_backbone"])
    sig = inspect.signature(backbone.__init__)
    model_kwargs = {}
    for parameter in sig.parameters:
        parameter = sig.parameters[parameter]
        # Skip self
        if parameter.name == "self":
            continue
        # Skip input_bands and output_bands parameters that are
        if parameter.name in ["input_bands", "output_bands"]:
            model_kwargs[parameter.name] = len(config_dict[parameter.name])
            continue
        if parameter.name not in config_dict["model"]:
            gs.warning(
                _(
                    "Parameter '{0}' missing in model configuration, using default {1}"
                ).format(parameter.name, parameter.default)
            )
            model_kwargs[parameter.name] = parameter.default
        else:
            if not isinstance(
                config_dict["model"][parameter.name], parameter.annotation
            ):
                gs.fatal(
                    _(
                        "Invalid data type for parameter '{0}': given {1}, required {2}"
                    ).format(
                        parameter.name,
                        config_dict["model"][parameter.name].__class__.__name__,
                        parameter.annotation.__name__,
                    )
                )
            model_kwargs[parameter.name] = config_dict["model"][parameter.name]

    # Check input and output description
    input_band_dict = {
        "order": {"type": int, "length": None, "content": None},
        "valid_range": {
            "type": (tuple, list, None),
            "length": 2,
            "content": (int, float, None),
        },
        "offset": {"type": (int, float), "length": None, "content": None},
        "scale": {"type": (int, float), "length": None, "content": None},
        "fill_value": {
            "type": (int, float, None),
            "length": None,
            "content": None,
        },  # None will be replaced by np.nan
    }
    required_keys = ["input_bands", "output_bands"]
    config_keys = {
        "input_bands": input_band_dict,
        "reference_bands": input_band_dict,
        "auxillary_bands": input_band_dict,
        "output_bands": {
            "valid_output_range": {
                "type": (tuple, list, None),
                "length": 2,
                "content": (int, float, None),
            },
            "title": {"type": str, "length": None, "content": None},
            "semantic_label": {"type": str, "length": None, "content": None},
            "keywords": {"type": (tuple, list), "length": None, "content": str},
            "standard_name": {"type": str, "length": None, "content": None},
            "description": {"type": (tuple, list, str), "length": None, "content": str},
            "units": {"type": str, "length": None, "content": None},
            "fill_value": {
                "type": (int, float, None),
                "length": None,
                "content": None,
            },  # None will be replaced by np.nan
            "offset": {"type": (int, float), "length": None, "content": None},
            "scale": {"type": (int, float), "length": None, "content": None},
            "classes": {"type": dict, "length": None, "content": None},
        },
    }
    for config_key in config_keys:
        if config_key not in config_dict:
            if config_key in required_keys:
                # Check if required key is present in section
                gs.fatal(
                    _("Key '{0}' missing in input config file {1}").format(
                        config_key, str(json_path)
                    )
                )
            else:
                continue
        if config_key in {"reference_bands", "auxillary_bands"}:
            model_kwargs["input_bands"] += len(config_dict[config_key])

        for config_sub_key in config_keys[config_key]:
            for band, band_description in config_dict[config_key].items():
                # Check if required sub-key is present in section
                if config_sub_key not in band_description:
                    gs.fatal(
                        _(
                            "Key '{0}' missing in section '{1}' for band {2} in input config file <{3}>"
                        ).format(config_sub_key, config_key, band, str(json_path))
                    )
                # Check if sub-key is of required data type
                type_mismatch = not_in_types(
                    band_description[config_sub_key],
                    config_keys[config_key][config_sub_key]["type"],
                )
                if type_mismatch:
                    gs.fatal(
                        _(
                            "Value for key '{0}' in section '{1}' is of data type '{2}' while {3} is required"
                        ).format(
                            config_sub_key,
                            config_key,
                            type_mismatch[0],
                            ", ".join(type_mismatch[1]),
                        )
                    )
                # Check if sub-key has the required length
                if (
                    config_keys[config_key][config_sub_key]["length"]
                    and band_description[config_sub_key]
                    and len(band_description[config_sub_key])
                    != config_keys[config_key][config_sub_key]["length"]
                ):
                    gs.fatal(
                        _(
                            "Key '{0}' in section '{1}' has to be of length {2}, given length for band {3} is {4}"
                        ).format(
                            config_sub_key,
                            config_key,
                            config_keys[config_key][config_sub_key]["length"],
                            band,
                            len(band_description[config_sub_key]),
                        )
                    )
                # Check if sub-key contains required elements
                if (
                    config_keys[config_key][config_sub_key]["content"]
                    and band_description[config_sub_key]
                ):
                    for idx, _key_element in enumerate(
                        band_description[config_sub_key]
                    ):
                        type_mismatch = not_in_types(
                            band_description[config_sub_key],
                            config_keys[config_key][config_sub_key]["type"],
                        )
                        if type_mismatch:
                            gs.fatal(
                                _(
                                    "Element {0} of iterable in key '{1}' in section '{2}' is of data type '{3}' while {4} is required"
                                ).format(
                                    idx + 1,
                                    config_sub_key,
                                    config_key,
                                    type_mismatch[0],
                                    ", ".join(type_mismatch[1]),
                                )
                            )

    return config_dict, backbone, model_kwargs
