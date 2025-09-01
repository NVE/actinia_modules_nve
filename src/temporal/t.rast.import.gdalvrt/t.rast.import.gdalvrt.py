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
# % description: Create a VRT (Virtual Raster Tile) from multiple raster files and import it to a STRDS.
# % keyword: gdal
# % keyword: vrt
# % keyword: COG
# % keyword: temporal
# % keyword: strds
# %end

# %option G_OPT_M_DIR
# % key: input
# % description: Name of input directory with raster files to create VRT from
# % required: yes
# %end

# %option
# % key: suffix
# % description: Suffix of files to include in VRT (default: .tif)
# % answer: tif
# % multiple: no
# % required: yes
# %end

# %option G_OPT_M_DIR
# % key: vrt_directory
# % description: Name of directory into which VRT-files are written (default=input)
# % required: no
# %end

# %option G_OPT_STRDS_OUTPUT
# %end

# %option
# % key: title
# % description: Title for the output STRDS (only used for new STRDS)
# % type: string
# % multiple: no
# % required: no
# %end

# %option
# % key: description
# % description: Description of the output STRDS (only used for new STRDS)
# % type: string
# % multiple: no
# % required: no
# %end

# %option G_OPT_F_INPUT
# % key: bands
# % description: JSON file with band configuration
# % type: string
# %end

# %option
# % key: basename
# % description: Basename for VRT and GRASS raster maps to create
# % type: string
# % multiple: no
# % required: no
# %end

# %option
# % key: start_time
# % description: Start time (ISO format) to register the VRT / GRASS raster maps with
# % type: string
# % multiple: no
# % required: yes
# %end

# %option
# % key: end_time
# % description: End time (ISO format) to register the VRT / GRASS raster maps with
# % type: string
# % multiple: no
# % required: no
# %end

# %option
# % key: file_pattern
# % description: File name pattern to import
# % type: string
# % multiple: no
# % guisection: Filter
# %end

# %option
# % key: memory
# % type: integer
# % required: no
# % multiple: no
# % label: Maximum memory to be used (in MB)
# % description: Cache size for raster rows
# % answer: 300
# %end

# %flag
# % key: l
# % description: Link the raster files using r.external
# % guisection: Settings
# %end

# %flag
# % key: f
# % description: Link the raster files in a fast way, without reading metadata using r.external
# % guisection: Settings
# %end

# %flag
# % key: e
# % description: Extend existing STRDS
# % guisection: Settings
# %end

# %rules
# % exclusive: -l,-f
# % required: -e,title
# % required: -e,description
# % collective: title,description
# %end

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import grass.lib.raster as libraster
import grass.script as gs
import grass.temporal as tgis
from grass.pygrass.gis import Mapset
from grass.pygrass.modules import Module
from grass.temporal.register import register_maps_in_space_time_dataset
from osgeo import gdal

if TYPE_CHECKING:
    from osgeo import gdal


def open_strds(
    strds: str,
    title: str | None = None,
    description: str | None = None,
    *,
    append: bool = False,
    overwrite: bool = False,
    mapset: str = Mapset().name,
) -> str:
    """Open a GRASS SpaceTimeRasterDataset (STRDS) for updating.

    Creates a new STRDS if it does not exist, or opens an existing one for appending data.

    Parameters
    ----------
    strds : str
        Name of the SpaceTimeRasterDataset to open or create.
    title : str | None
        Title of the STRDS to create
    description : str | None
        Description of the STRDS to create
    append : bool, optional
        If True, opens the STRDS for appending data. Defaults to False.
    overwrite : bool, optional
        If True, overwrites the existing STRDS if it exists. Defaults to False.
    mapset : str
        Name of the current mapset

    Returns
    -------
    tgis.SpaceTimeRasterDataset

    """
    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = strds if "@" in strds else f"{strds}@{mapset}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if tgis_strds.is_in_db() and not overwrite:
        gs.fatal(
            _(
                "Output STRDS <{}> exists."
                " Use --overwrite with or without -e to modify the existing STRDS.",
            ).format(strds),
        )
    if not tgis_strds.is_in_db() or (overwrite and not append):
        Module(
            "t.create",
            output=strds,
            type="strds",
            temporaltype="absolute",
            title=title,
            description=description,
            verbose=True,
        )
    return strds_long_name


def get_gdal_band_color_interpretation() -> dict:
    """Get GDAL band color interpretation as a dictionary."""
    minimal_gdal_version = 3100000
    get_gdal_band_colors = {
        "Undefined": gdal.GCI_Undefined,
        "Greyscale": gdal.GCI_GrayIndex,
        "Paletted": gdal.GCI_PaletteIndex,  # (see associated color table)
        "Red": gdal.GCI_RedBand,  # RGBA image, or red spectral band [0.62 - 0.69 um]
        "Green": gdal.GCI_GreenBand,  # RGBA image, or green spectral band [0.51 - 0.60 um]
        "Blue": gdal.GCI_BlueBand,  # RGBA image, or blue spectral band [0.45 - 0.53 um]
        "Alpha": gdal.GCI_AlphaBand,  # (0=transparent, 255=opaque)
        "Hue": gdal.GCI_HueBand,  # HLS image
        "Saturation": gdal.GCI_SaturationBand,  # HLS image
        "Lightness": gdal.GCI_LightnessBand,  # HLS image
        "Cyan": gdal.GCI_CyanBand,  # CMYK image
        "Magenta": gdal.GCI_MagentaBand,  # CMYK image
        "Yellow": gdal.GCI_YellowBand,  # CMYK image, or yellow spectral band [0.58 - 0.62 um]
        "Black": gdal.GCI_BlackBand,  # CMYK image
        "Y": gdal.GCI_YCbCr_YBand,  # Luminance
        "Cb": gdal.GCI_YCbCr_CbBand,  # Chroma
        "Cr": gdal.GCI_YCbCr_CrBand,  # Chroma
    }
    if int(gdal.VersionInfo()) >= minimal_gdal_version:
        get_gdal_band_colors.update(
            {
                "Panchromatic": gdal.GCI_PanBand,  # [0.40 - 1.00 um]
                "Coastal": gdal.GCI_CoastalBand,  # [0.40 - 0.45 um]
                "Red-edge": gdal.GCI_RedEdgeBand,  # [0.69 - 0.79 um]
                "Near-InfraRed (NIR)": gdal.GCI_NIRBand,  # [0.75 - 1.40 um]
                "Short-Wavelength InfraRed (SWIR)": gdal.GCI_SWIRBand,  # [1.40 - 3.00 um]
                "Mid-Wavelength InfraRed (MWIR)": gdal.GCI_MWIRBand,  # [3.00 - 8.00 um]
                "Long-Wavelength InfraRed (LWIR)": gdal.GCI_LWIRBand,  # [8.00 - 15 um]
                "Thermal InfraRed (TIR)": gdal.GCI_TIRBand,  # (MWIR or LWIR) [3 - 15 um]
                "Other infrared": gdal.GCI_OtherIRBand,  # [0.75 - 1000 um]
                # "Reserved value": gdal.GCI_IR_Reserved_1,  # Do not set it !
                # "Reserved value": gdal.GCI_IR_Reserved_2,  # Do not set it !
                # "Reserved value": gdal.GCI_IR_Reserved_3,  # Do not set it !
                # "Reserved value": gdal.GCI_IR_Reserved_4,  # Do not set it !
                "Synthetic Aperture Radar (SAR) Ka": gdal.GCI_SAR_Ka_Band,  # [0.8 - 1.1 cm / 27 - 40 GHz]
                "Synthetic Aperture Radar (SAR) K": gdal.GCI_SAR_K_Band,  # [1.1 - 1.7 cm / 18 - 27 GHz]
                "Synthetic Aperture Radar (SAR) Ku": gdal.GCI_SAR_Ku_Band,  # [1.7 - 2.4 cm / 12 - 18 GHz]
                "Synthetic Aperture Radar (SAR) X": gdal.GCI_SAR_X_Band,  # [2.4 - 3.8 cm / 8 - 12 GHz]
                "Synthetic Aperture Radar (SAR) C": gdal.GCI_SAR_C_Band,  # [3.8 - 7.5 cm / 4 - 8 GHz]
                "Synthetic Aperture Radar (SAR) S": gdal.GCI_SAR_S_Band,  # [7.5 - 15 cm / 2 - 4 GHz]
                "Synthetic Aperture Radar (SAR) L": gdal.GCI_SAR_L_Band,  # [15 - 30 cm / 1 - 2 GHz]
                "Synthetic Aperture Radar (SAR) P": gdal.GCI_SAR_P_Band,  # [30 - 100 cm / 0.3 - 1 GHz]
                "SAR Reserved value": gdal.GCI_SAR_Reserved_1,  # Do not set it !
                # "SAR Reserved value": gdal.GCI_SAR_Reserved_2,  # Do not set it !
                "Max current value": gdal.GCI_Max,  # (equals to GCI_SAR_Reserved_2 currently)
            },
        )
    return get_gdal_band_colors


def build_vrt(
    raster_directory: Path,
    vrt_directory: Path,
    band_template: dict[str, str],
    *,
    vrt_name: str | None = None,
    raster_file_pattern: str = "*.tif",
    multiband: bool = True,
) -> tuple[str, dict]:
    """Build a VRT file for GDAL readable raster data in a directory.

    :param raster_directory: Path to the directory containing GTIFF files.
    :param vrt_directory: Path to the directory where the VRT file
                          will be saved.
    :param band_template: Dictionary with expected band names.
    :param vrt_name: Name of the VRT dataset to produce.
    :multiband: Contain the input rasters multiple bands

    Assumes a specific naming convention and that:
    1) the suffix of the GTIFF files is .tif if no pattern is provided,
    2) that the the end of the file name contains
       the name / id of the semantic label / band, and
    3) that the input GTIFFs are single band files.

    """
    tiffs = list(raster_directory.glob(raster_file_pattern))

    if not vrt_name:
        vrt_name = Path(os.path.commonprefix(tiffs)).stem
    vrt_path = vrt_directory / f"{vrt_name}.vrt"
    ds = gdal.BuildVRT(str(vrt_path), tiffs, separate=not multiband)

    raster_bands = ds.RasterCount
    if len(band_template) != raster_bands:
        gs.fatal(
            _(
                "Band configuration contains {config_bands} bands, VRT file {vrt_bands}",
            ).format(config_bands=len(band_template), vrt_bands=raster_bands),
        )
        ds = None
        vrt_path.unlink()
    for bid, band_tuple in enumerate(band_template.items()):
        # GDAL bands are 1-indexed
        band = ds.GetRasterBand(bid + 1)
        # Set band name
        band.SetDescription(band_tuple[0])
        # Set band color interpretation
        color_interpretation = GDAL_BAND_COLOR_INTERPRETATION.get(
            band_tuple[1],
            gdal.GCI_Undefined,
        )
        band.SetColorInterpretation(color_interpretation)
        # Here we could set band metadata
        # See:
        # https://gdal.org/en/stable/drivers/raster/gtiff.html#metadata
        # https://gdal.org/en/stable/user/raster_data_model.html#imagery-domain-remote-sensing

    ds = None

    return vrt_path


def import_vrt(
    vrt: Path,
    band_template: dict[str, str],
    *,
    mapset: str = Mapset().name,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
    link: bool = True,
    fast: bool = False,
) -> list[str]:
    """Import a VRT file into GRASS GIS as an external raster map."""
    register_strings = []
    for idx, band_id in enumerate(band_template):
        output_name = f"{vrt.stem}.{band_id}"
        Module(
            "r.external" if link or fast else "r.in.gdal",
            input=str(vrt),
            output=output_name,
            band=idx + 1,
            flags="re" if fast else "e",
            overwrite=gs.overwrite(),
        )
        semantic_label = band_id
        register_string = (
            f"{output_name}@{mapset}|{start_time.strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        if end_time:
            register_string += f"|{end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
        if semantic_label:
            register_string += f"|{semantic_label}"
        register_strings.append(register_string)
    return register_strings


GDAL_BAND_COLOR_INTERPRETATION = get_gdal_band_color_interpretation()


def main() -> None:
    """Import using Sentinel2Importer."""
    # Get bands configuration info
    try:
        band_config = json.loads(Path(options["bands"]).read_text(encoding="UTF8"))
    except json.JSONDecodeError:
        gs.fatal(_("Band configuration file is not a valid JSON file."))

    # Check that band IDs are valid semantic labels in GRASS
    for band_id in band_config:
        if libraster.Rast_legal_semantic_label(band_id) is False:
            gs.fatal(
                _(
                    'Band ID "{band_id}" is not a valid semantic label.',
                ).format(band_id=band_id),
            )

    # Create output directory if needed
    input_dir = Path(options["input"])

    # Create output directory if needed
    output_dir = Path(options["vrt_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for dt in ("end_time", "start_time"):
            if dt == "end_time" and not options[dt]:
                options[dt] = None
                continue
            options[dt] = datetime.fromisoformat(options[dt].replace("Z", ""))
    except ValueError:
        gs.fatal(
            _("Input for '{}' is not a valid ISO format. Got {}").format(
                dt, options[dt],
            ),
        )

    vrt = build_vrt(
        input_dir,
        output_dir,
        band_config,
        vrt_name=options["basename"] or None,
        raster_file_pattern=options["file_pattern"],
    )

    register_strings = import_vrt(
        vrt,
        band_config,
        start_time=options["start_time"],
        end_time=options["end_time"],
        link=flags["l"] or flags["f"],
        fast=flags["f"],
    )

    # Initialize TGIS
    tgis.init()
    strds = open_strds(
        options["output"],
        options["title"],
        options["description"],
        append=flags["e"],
        overwrite=gs.overwrite(),
    )

    gs.verbose(_("Registering imported maps in STRDS..."))
    # Create temporal register file
    map_file = Path(gs.tempfile())
    map_file.write_text(
        "\n".join({r_s for r_s in register_strings if r_s is not None}),
        encoding="UTF8",
    )

    register_maps_in_space_time_dataset(
        "raster",
        strds,
        file=map_file,
        update_cmd_list=False,
        fs="|",
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
