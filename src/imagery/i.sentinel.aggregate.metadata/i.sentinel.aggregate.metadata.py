#!/usr/bin/env python3
"""MODULE:      t.rast.import.gdalvrt
AUTHOR(S):      Stefan Blumentrath
PURPOSE:        Create a VRT (Virtual Raster Tile) from multiple raster files and import it to a STRDS
COPYRIGHT:      (C) 2025 by Stefan Blumentrath, NVE
                and the GRASS development team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

# %Module
# % description: Aggregate Sentinel product metadata from several tiles or scenes.
# % keyword: imagery
# % keyword: sentinel
# % keyword: sentinel-1
# % keyword: sentinel-2
# % keyword: sentinel-3
# % keyword: metadata
# %end

# %option G_OPT_M_DIR
# % key: input
# % description: Name of input directory with json files to aggregate
# % required: yes
# %end

# %option
# % key: product_type
# % description: Name of the type of Sentinel-products the metadata belongs to
# % options: S1GRDH,S2MSIL1C,S2MSIL2A,S3OLCIL1B,S3SLSTRL1B
# % type: string
# % required: yes
# %end

# %option G_OPT_F_OUTPUT
# % required: no
# %end

# %option
# % key: file_pattern
# % description: File name pattern of json files to aggregate (regular expression)
# % type: string
# % required: no
# % answer: **/*.json
# % guisection: Filter
# %end


import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import grass.script as gs


def aggregate_metadata(json_files: Path, product_type: str = "S2_MSI_L1C") -> dict:
    """Aggregate metadata JSON files."""
    metadata_keys = {
        "PRODUCT_START_TIME": datetime(1, 1, 1, 0, 0, 0),  # '2025-08-11T10:37:01.024Z',
        "PRODUCT_STOP_TIME": datetime.now()
        + timedelta(days=365 * 1000),  # '2025-08-11T10:37:01.024Z',
        "PRODUCT_URI": set(),  # 'S2A_MSIL2A_20250811T103701_N0511_R008_T33WXT_20250811T173718.SAFE',
        "PROCESSING_LEVEL": set(),  # 'Level-2A',
        "PRODUCT_TYPE": set(),  # 'S2MSI2A',
        "PROCESSING_BASELINE": set(),  # '05.11',
        "PRODUCT_DOI": set(),  # 'https://doi.org/10.5270/S2_-znk9xsj',
        "GENERATION_TIME": [
            datetime(1, 1, 1, 0, 0, 0),
            datetime.now() + timedelta(days=365 * 1000),
        ],  # '2025-08-11T17:37:18.000000Z',
        "PREVIEW_IMAGE_URL": set(),  # 'Not applicable',
        "PREVIEW_GEO_INFO": set(),  # 'Not applicable',
        "SPACECRAFT_NAME": set(),  # 'Sentinel-2A',
        "DATATAKE_TYPE": set(),  # 'INS-NOBS',
        "DATATAKE_SENSING_START": [
            datetime(1, 1, 1, 0, 0, 0),
            datetime.now() + timedelta(days=365 * 1000),
        ],  # '2025-08-11T10:37:01.024Z',
        "SENSING_ORBIT_NUMBER": set(),  # '8',
        "SENSING_ORBIT_DIRECTION": set(),  # 'DESCENDING',
        "PRODUCT_FORMAT": set(),  # 'SAFE_COMPACT',
        "IMAGE_FILE": set(),  # 'GRANULE/L2A_T33WXT_A052944_20250811T103955/IMG_DATA/R60m/T33WXT_20250811T103701_SCL_60m',
        "CLOUDY_PIXEL_OVER_LAND_PERCENTAGE": 0.0,  # '77.333683',
        "CLOUDY_PIXEL_PERCENTAGE": 0.0,  # '74.293292',
        "DEGRADED_MSI_DATA_PERCENTAGE": 0.0,  # '0.002200',
        "MEAN_SUN_ZENITH_ANGLE": 0.0,  # 54.6655024578069,
        "MEAN_SUN_AZIMUTH_ANGLE": 0.0,  # 177.779131794776,
        "FORMAT_CORRECTNESS": set(),  # 'PASSED',
        "GENERAL_QUALITY": set(),  # 'PASSED',
        "GEOMETRIC_QUALITY": set(),  # 'PASSED',
        "RADIOMETRIC_QUALITY": set(),  # 'PASSED',
        "SENSOR_QUALITY": set(),  # 'PASSED'
    }
    if product_type == "S2_MSI_L2A":
        metadata_keys.update(
            {
                "NODATA_PIXEL_PERCENTAGE": 0.0,  # '0.000000',
                "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE": 0.0,  # '0.000000',
                "CAST_SHADOW_PERCENTAGE": 0.0,  # '0.442563',
                "CLOUD_SHADOW_PERCENTAGE": 0.0,  # '2.054479',
                "VEGETATION_PERCENTAGE": 0.0,  # '6.534617',
                "NOT_VEGETATED_PERCENTAGE": 0.0,  # '1.082438',
                "WATER_PERCENTAGE": 0.0,  # '13.964827',
                "UNCLASSIFIED_PERCENTAGE": 0.0,  # '1.513396',
                "MEDIUM_PROBA_CLOUDS_PERCENTAGE": 0.0,  # '13.894749',
                "HIGH_PROBA_CLOUDS_PERCENTAGE": 0.0,  # '58.243871',
                "THIN_CIRRUS_PERCENTAGE": 0.0,  # '2.154681',
                "SNOW_ICE_PERCENTAGE": 0.0,  # '0.114386',
                "RADIATIVE_TRANSFER_ACCURACY": 0.0,  # '0.0',
                "WATER_VAPOUR_RETRIEVAL_ACCURACY": 0.0,  # '0.0',
                "AOT_RETRIEVAL_ACCURACY": 0.0,  # '0.0',
                "AOT_RETRIEVAL_METHOD": set(),  # 'CAMS',
                "GRANULE_MEAN_AOT": 0.0,  # '0.070595',
                "GRANULE_MEAN_WV": 0.0,  # '1.314649',
                "OZONE_SOURCE": set(),  # 'AUX_ECMWFT',
                "OZONE_VALUE": 0.0,  # '302.702650',
                "L2A_QUALITY": set(),  # 'PASSED',
            },
        )
    valid_data_total = 0.0
    for json_file in json_files:
        try:
            meta_data = json.loads(json_file.read_text(encoding="utf-8"))
            meta_data = meta_data.get("metadata")
            meta_data = meta_data.get("product_metadata")
            print(product_type)
            valid_data_percent = (
                1
                if product_type == "S2_MSI_L1C"
                else 100.0 - float(meta_data.get("NODATA_PIXEL_PERCENTAGE"))
            )
            for key, val in metadata_keys.items():
                print(meta_data)
                print(key in meta_data)
                if key in meta_data:
                    print(key, val, type(val))
                    if isinstance(val, float):
                        metadata_keys[key] += (
                            float(meta_data.get(key)) * valid_data_percent
                        )
                    elif isinstance(val, set):
                        metadata_keys[key].add(meta_data.get(key))
                    elif isinstance(val, datetime):
                        print(
                            metadata_keys[key],
                            datetime.fromisoformat(meta_data.get(key).replace("Z", "")),
                        )
                        if "START" in key:
                            metadata_keys[key] = max(
                                metadata_keys[key],
                                datetime.fromisoformat(
                                    meta_data.get(key).replace("Z", ""),
                                ),
                            )
                        else:
                            print(
                                metadata_keys[key],
                                datetime.fromisoformat(
                                    meta_data.get(key).replace("Z", ""),
                                ),
                            )
                            metadata_keys[key] = min(
                                metadata_keys[key],
                                datetime.fromisoformat(
                                    meta_data.get(key).replace("Z", ""),
                                ),
                            )
                    elif isinstance(val, list):
                        metadata_keys[key][0] = max(
                            metadata_keys[key][0],
                            datetime.fromisoformat(meta_data.get(key).replace("Z", "")),
                        )
                        metadata_keys[key][1] = min(
                            metadata_keys[key][1],
                            datetime.fromisoformat(meta_data.get(key).replace("Z", "")),
                        )
                else:
                    print(key, val)

            valid_data_total += valid_data_percent
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_file}: {e}")

    for key, val in metadata_keys.items():
        if isinstance(val, float):
            metadata_keys[key] /= valid_data_total
        elif isinstance(val, datetime):
            metadata_keys[key] = val.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(val, set):
            if len(val) == 1:
                metadata_keys[key] = next(iter(val))
            else:
                metadata_keys[key] = ",".join(sorted(val))
        elif isinstance(val, list):
            if val[0] != val[1]:
                metadata_keys[key] = "/".join(
                    d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in val
                )
            else:
                metadata_keys[key] = val[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    return metadata_keys


def main() -> None:
    """Aggregate Sentinel scene metadata."""
    # Get bands configuration info
    input_dir = Path(options["input"])
    file_pattern = options["pattern"]
    jsons = list(
        input_dir.glob("*.json") if not file_pattern else input_dir.glob(file_pattern)
    )
    if len(jsons) == 0:
        gs.fatal(
            _("No JSON files found in <{}> with file pattern <{}>").format(
                input_dir, file_pattern
            )
        )

    metadata = aggregate_metadata(
        jsons,
        product_type=options["product_type"],
    )
    if not options["output"]:
        print(metadata)
    else:
        output_file = Path(options["output"])
        try:
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except OSError as e:
            gs.fatal(
                _("Unable to write to output file <{}>: {}").format(output_file, e)
            )


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    try:
        from osgeo import gdal

        gdal.UseExceptions()
    except ImportError as e:
        gs.fatal(_("Unable to load GDAL Python bindings: {}").format(e))

    sys.exit(main())
