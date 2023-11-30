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

# import json
# from pathlib import Path

import grass.script as gs

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.autograd import Variable
except ImportError:
    gs.fatal(("Could not import pytorch. Please make sure it is installed."))
import numpy as np

try:
    from pytorchlib.backbones import UNetV2
except ImportError:
    gs.fatal(
        ("Could not import included unet library. Please check the addon installation.")
    )


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
        return torch.Tensor.cpu(torch_tensor).numpy()
    return torch_tensor.numpy()


def validate_config(config_dict):
    required_keys = {
        "model": {
            "type": str,
            "n_classes": int,
            "depth": int,
            "start_filts": int,
            "up_mode": str,
            "merge_mode": str,
            "partial_conv": bool,
            "use_bn": bool,
            "activation_func": str,
        },
        "input_bands": {
            "valid_range": (tuple, list),
            "offset": (int, float),
            "scale": (int, float),
        },
        "output_bands": {
            "valid_output_range": list,
            "title": str,
            "semantic_label": str,
            "description": str,
            "units": str,
        },
    }
    for config_key in required_keys:
        if config_key not in config_dict:
            gs.fatal(
                _("Key '{0}' missing in input config file {1}").format(
                    config_key, str(json_path)
                )
            )
        for config_sub_key in required_keys[config_key]:
            if config_key == "model":
                if config_sub_key not in config_dict[config_key]:
                    gs.fatal(
                        _(
                            "Key '{0}' missing in section '{1}' input config file {2}"
                        ).format(config_sub_key, config_key, str(json_path))
                    )
                required_data_type = required_keys[config_key][config_sub_key]
                actual_data_type = config_dict[config_key][config_sub_key]
                if not isinstance(actual_data_type, required_data_type):
                    gs.fatal(
                        _(
                            "Value for key '{0}' in section '{1}' is of data type '{2}' while '{3}' is required"
                        ).format(
                            config_sub_key,
                            config_key,
                            actual_data_type,
                            required_data_type,
                        )
                    )
            elif config_key == "input_bands":
                for input_band, description in config_dict[config_key].items():
                    if not isinstance(description[0], int):
                        gs.fatal(
                            _("Index for input band '{0}' needs to be numeric").format(
                                input_band
                            )
                        )

                    if config_sub_key not in description[1]:
                        gs.fatal(
                            _(
                                "Key '{0}' missing in section '{1}' for input band {2} input config file {3}"
                            ).format(
                                config_sub_key, config_key, input_band, str(json_path)
                            )
                        )
                    required_data_type = required_keys[config_key][config_sub_key]
                    actual_data_type = description[1][config_sub_key]
                    if not isinstance(
                        actual_data_type, required_data_type[0]
                    ) and not isinstance(actual_data_type, required_data_type[1]):
                        gs.fatal(
                            _(
                                "Value for key '{0}' in section '{1}' is of data type '{2}' while '{3}' is required"
                            ).format(
                                config_sub_key,
                                config_key,
                                actual_data_type,
                                required_data_type,
                            )
                        )
            elif config_key == "output_bands":
                for output_band, description in config_dict[config_key].items():
                    if config_sub_key not in description:
                        gs.fatal(
                            _(
                                "Key '{0}' missing in section '{1}' for input band {2} input config file {3}"
                            ).format(
                                config_sub_key, config_key, output_band, str(json_path)
                            )
                        )
                    required_data_type = required_keys[config_key][config_sub_key]
                    actual_data_type = description[config_sub_key]
                    if not isinstance(actual_data_type, required_data_type):
                        gs.fatal(
                            _(
                                "Value for key '{0}' in section '{1}' of output band {2} is of data type '{3}' while '{4}' is required"
                            ).format(
                                config_sub_key,
                                config_key,
                                output_band,
                                actual_data_type,
                                required_data_type,
                            )
                        )

    return 0
