#!/usr/bin/env python3

"""
 MODULE:       i.satskred
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Run avalanche detection from Sentinel-1 imagery using satskred
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
 - unclear parallelization (ask NORCE)
 - Handle CRS (don't hard-code, may require adjustment in satskred code)
 - allow only geocoding
 - support zipped S1 archives

 time satskred init \
  --config /hdata/SatSkredTestArea/.satskredconf.json \
  --woodpecker-config /hdata/SatSkredTestArea/.woodpeckerconf.json \
  --avaldet-config /hdata/SatSkredTestArea/.avaldet.json \
  --projname UTM33N \
  /hdata/SatSkredTestArea/lavangsdalen \
  660179.462 7718470.946 671390.957 7705610.702

time satskred run \
  --config /hdata/SatSkredTestArea/.satskredconf.json \
  --woodpecker-config /hdata/SatSkredTestArea/.woodpeckerconf.json \
  --avaldet-config /hdata/SatSkredTestArea/.avaldet.json \
  --logfile /tmp/satskred.log --loglevel 20 \
  /hdata/SatSkredTestArea/lavangsdalen

"""

# %module
# % description: Run avalanche detection from Sentinel-1 imagery using satskred
# % keyword: raster
# % keyword: imagery
# % keyword: copernicus
# % keyword: sentinel
# % keyword: satellite
# % keyword: radar
# % keyword: satskred
# % keyword: avalanche
# % keyword: snow
# %end

# %option G_OPT_M_DIR
# %key: input
# % description: Input directory with Sentinel-1 imagery
# %end

# %option
# %key: elevation
# % type: string
# % required: yes
# % multiple: no
# % description: Digital elevation model to use for geocoding (either a path to a GeoTiff or a linked raster map)
# %end

# %option G_OPT_M_DIR
# % key: output_directory
# % required: no
# % answer: ./
# % description: Name for output directory where satskred results are stored (default: ./)
# % label: Directory where satskred results are stored
# %end

# %option G_OPT_M_DIR
# % key: mask_directory
# % required: yes
# % description: Name for output directory where masks for avalanche detection are stored
# % label: Directory where masks for avalanche detection are stored
# %end

# %option
# % key: mask_suffix
# % type: string
# % required: no
# % answer: tif
# % description: Suffix used for files with runout masks
# %end

# %option
# % key: mask_exclude
# % type: integer
# % required: yes
# % multiple: yes
# % description: Comma separated list of category values in mask to exclude from avalanche detection
# %end

# %option
# % key: mask_min_valid
# % type: integer
# % required: yes
# % multiple: no
# % description: Minimum valid category value in mask (values >= mask_min_valid are incuded in avalanche detection)
# %end

# %option
# % key: start
# % type: string
# % required: no
# % description: Start date of time frame to compute (required format: YYYY-MM-DD)
# % label: Start date of time frame to compute (required format: YYYY-MM-DD)
# %end

# %option
# % key: end
# % type: string
# % required: no
# % description: End date of time frame to compute (required format: YYYY-MM-DD)
# % label: End date of time frame to compute (required format: YYYY-MM-DD)
# %end


import json
import shutil
import sys

from datetime import datetime

# from multiprocessing import Pool
from pathlib import Path

# from subprocess import PIPE

import grass.script as gs


# Configuration
def write_config(
    directory=Path("./"),
    logfile="satskred.log",
    loglevel=20,
    sar="/hdata/SatSkredTestArea/S1/*.SAFE",
    reporting="./",
    dem=None,
    reference_height="geoid",
    masks=None,
    mask_name=None,
    mask_hard=True,
    mask_excluded_values=(0, 3),
):
    """Write configuration files for satskred
    according to user input"""
    config = {
        "satskredconf": directory / ".satskredconf.json",
        "woodpeckerconf": directory / ".woodpeckerconf.json",
        "avaldetconf": directory / ".avaldetconf.json",
    }

    python_bin = shutil.which("python") or shutil.which("python3")
    if not python_bin:
        gs.fatal(_("python / python3 not found on PATH"))

    # Write satskred configuration
    config["satskredconf"].write_text(
        json.dumps(
            {
                "python": python_bin,
                "condaenv": "",
                "processor": "woodpecker",
                "logfile": logfile,
                "loglevel": loglevel,
                "projectdir": "",
                "datadir": "",
                "node": "",
                "data_sources": {
                    "sar": {"source_spec": {"type": "dir", "source": sar}},
                    "reporting": {"source_spec": {"type": "dir", "source": reporting}},
                    "dem": {
                        "source_spec": {"type": "dir", "source": dem},
                        "reference_height": reference_height,
                    },
                    "masks": [
                        {
                            "source_spec": {
                                "type": "dir",
                                "source": masks,
                            },
                            "name": mask_name,
                            "hard": mask_hard,
                            "excluded_values": mask_excluded_values,
                        }
                    ],
                },
            },
            indent=2,
        )
    )

    # Write woodpecker configuration
    config["woodpeckerconf"].write_text(
        json.dumps(
            {
                "geocode": 1,
                "output_rgb_tif": 1,
                "output_rgb_jpg": 1,
                "output_detection_tif": 1,
                "output_detection_shp": 1,
                "output_aggregation": 0,
                "dask": {"enabled": False, "client_kwargs": {"n_workers": 10}},
            },
            indent=2,
        )
    )

    # Write avaldet configuration
    config["avaldetconf"].write_text(
        json.dumps(
            {
                "edgefr": 0.3,
                "anomthr": 4,
                "classchfr": 0.5,
                "min_size": 15,
                "max_size": 400,
                "min_backscatter": -40,
                "min_valid_mask_pixels": 50,
                "min_edge_segment_fraction": 0.001,
                "dog_std_0": 1,
                "dog_std_1": 18,
                "n_slices": 10,
                "block_size": [1500, 1500],
                "border_size": [200, 200],
            },
            indent=2,
        )
    )

    return config


def main():
    """Do the main work"""

    # Check if satskred is available
    if not shutil.which("satskred"):
        gs.fatal(_("satskred commandline tool not found on current PATH"))

    dem = options[
        "elevation"
    ]  # "/hdata/SatSkredTestArea/dtm/DTM10_20200629_Lavangsdalen.tif"

    output_directory = Path(options["output_directory"])
    input_directory = Path(options["input"])

    if not input_directory.exists():
        gs.fatal(_("Input directoy {} not found").format(str(input_directory)))
    mask_directory = Path(options["mask_directory"])

    if not mask_directory.exists():
        gs.fatal(
            _("Directoy {} with runout masks not found").format(str(mask_directory))
        )

    temp_dir = Path(gs.tempdir())
    log_level = 20
    log_file = str(
        temp_dir / f"i_satskred_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    )

    satskred_config = write_config(
        directory=temp_dir,
        logfile=log_file,
        loglevel=log_level,
        sar=str(input_directory / "*.SAFE"),
        reporting=str(temp_dir),
        dem=dem,
        reference_height="geoid",
        masks=str(mask_directory / f"*.{options['mask_suffix']}"),
        mask_name="runoutmask",
        mask_hard=True,
        mask_excluded_values=[0, 3],
    )

    region = gs.parse_command("g.region", flags="ug")
    west, north, east, south = region["w"], region["n"], region["e"], region["s"]

    config_list = [
        "--config",
        satskred_config["satskredconf"],
        "--woodpecker-config",
        satskred_config["woodpeckerconf"],
        "--avaldet-config",
        satskred_config["avaldetconf"],
    ]
    region_list = list(map(str, [west, north, east, south]))

    if not output_directory.exists():
        gs.info(_("Initializing input region {}").format(output_directory.name))
        satskred_command = ["satskred", "init"] + config_list
        gs.call(
            ["satskred", "init"]
            + config_list
            + ["--projname", "UTM33N", str(output_directory)]
            + region_list
        )

    satskred_command = ["satskred", "run"] + config_list
    if options["start"]:
        satskred_command.extend(["--starttime", options["start"]])
    if options["end"]:
        satskred_command.extend(["--stoptime", options["end"]])
    satskred_command.extend(
        ["--logfile", log_file, "--loglevel", str(log_level), str(output_directory)]
    )
    gs.call(satskred_command)


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports

    # try:
    #     from osgeo import gdal, ogr, osr
    # except ImportError:
    #     gs.fatal(
    #         _(
    #             "Can not import GDAL python bindings. Please install it with 'pip install GDAL==${GDAL_VERSION}'"
    #         )
    #     )

    sys.exit(main())
