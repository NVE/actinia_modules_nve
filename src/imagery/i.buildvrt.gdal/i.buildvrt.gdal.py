#!/usr/bin/env python3
"""MODULE:      i.buildvrt.gdal
AUTHOR(S):      Stefan Blumentrath
PURPOSE:        Create a Virtual Raster Tile (VRT) from multiple raster files
COPYRIGHT:      (C) 2025 by Stefan Blumentrath, NVE
                and the GRASS development team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""

# %Module
# % description: Create a VRT (Virtual Raster Tile) from multiple raster files and import it to a STRDS.
# % keyword: imagery
# % keyword: raster
# % keyword: gdal
# % keyword: vrt
# %end

# %option G_OPT_M_DIR
# % key: input
# % description: Name of input directory with raster files to create VRT from
# % required: yes
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % description: Path to the VRT file to be written
# %end

# %option G_OPT_F_INPUT
# % key: bands
# % description: JSON file with band configuration
# % type: string
# %end

# %option
# % key: file_pattern
# % description: File name pattern to import
# % type: string
# % multiple: no
# % guisection: Filter
# %end

# %option
# % key: data_type
# % description: GDAL data type of the output VRT (if data type of input rasters differs
# % options: Byte,Int8,UInt16,Int16,UInt32,Int32,UInt64,Int64,CInt16,CInt32,Float32,Float64,CFloat32,CFloat64
# % type: string
# % required: no
# % multiple: no
# % guisection: Settings
# %end

# %option
# % key: resolution
# % description: Resolution of the output VRT (if resolution of input rasters differs) can be an integer for the target resolution or one of 'average', 'highest', 'lowest'
# % type: string
# % answer: highest
# % required: no
# % multiple: no
# % guisection: Settings
# %end

# %option
# % key: memory
# % type: integer
# % required: no
# % multiple: no
# % label: Maximum memory to be used (in MB)
# % description: Cache size for raster rows
# % answer: 2048
# %end

# %flag
# % key: s
# % description: Stack input raster files as bands in VRT
# % guisection: Settings
# %end

import json
import sys
from pathlib import Path

import grass.lib.raster as libraster
import grass.script as gs


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
                # "SAR Reserved value": gdal.GCI_SAR_Reserved_1,  # Do not set it !
                # "SAR Reserved value": gdal.GCI_SAR_Reserved_2,  # Do not set it !
                "Max current value": gdal.GCI_Max,  # (equals to GCI_SAR_Reserved_2 currently)
            },
        )
    return get_gdal_band_colors


def build_vrt(
    raster_directory: Path,
    vrt_path: Path,
    band_template: dict[str, str],
    *,
    raster_file_pattern: str = "*.tif",
    multiband: bool = True,
    data_type: str | None = None,
    resolution: int | str = "highest",
) -> None:
    """Build a VRT file for GDAL readable raster data in a directory.

    :param raster_directory: Path to the directory containing GTIFF files.
    :param vrt_path: Path to the VRT file to produce.
    :param band_template: Dictionary with expected band names and GDAL colors.
    :param vrt_name: Name of the VRT dataset to produce.
    :param raster_file_pattern:
    :param multiband: Input rasters contain multiple bands
    :param data_type: GDAL data type name
    :param resolution: Resolution to use for the VRT. Can be determined from
                       raster files

    If multiband is False, each input raster is added as a separate
    band into the VRT. Otherwise, files are handled as tiles of a
    mosaic instead. Thus all input files need to have the same number
    of bands.

    """
    gdal_band_color_interpretation = get_gdal_band_color_interpretation()
    tiffs = raster_directory.glob(raster_file_pattern)

    # Create intermediate VRT to harmonize input if needed
    checked_tiffs = []
    for tiff in tiffs:
        ds = gdal.Open(tiff)
        x_res, y_res = abs(ds.GetGeoTransform()[1]), abs(ds.GetGeoTransform()[5])
        band = ds.GetRasterBand(1)
        dtype = gdal.GetDataTypeName(band.DataType)
        kwargs = {
            "format": "VRT",
            "outputType": gdal.GetDataTypeByName(data_type),
        }
        resolution_differs = x_res != resolution or y_res != resolution
        if isinstance(resolution, int):
            resolution_differs = x_res != resolution or y_res != resolution
            if resolution_differs:
                kwargs["xRes"] = resolution
                kwargs["yRes"] = resolution
        if dtype != data_type or resolution_differs:
            vrt = gdal.Translate(tiff.with_suffix(".vrt"), tiff, **kwargs)
            # Write and close dataset
            vrt.FlushCache()
            vrt = None
            checked_tiffs.append(str(tiff.with_suffix(".vrt")))
        else:
            checked_tiffs.append(str(tiff))
    checked_tiffs.sort()
    kwargs = {"separate": not multiband, "strict": False}

    if isinstance(resolution, str):
        kwargs["resolution"] = resolution
    ds = gdal.BuildVRT(str(vrt_path), checked_tiffs, **kwargs)

    raster_bands = ds.RasterCount
    if len(band_template) != raster_bands:
        ds = None
        vrt_path.unlink()
        gs.fatal(
            _(
                "Band configuration contains {config_bands} bands. "
                "VRT file only contains {vrt_bands} bands.",
            ).format(config_bands=len(band_template), vrt_bands=raster_bands),
        )
    for bid, band_tuple in enumerate(band_template.items()):
        # GDAL bands are 1-indexed
        band = ds.GetRasterBand(bid + 1)
        # Set band name
        band.SetDescription(band_tuple[0])
        # Set band color interpretation
        color_interpretation = gdal_band_color_interpretation.get(
            band_tuple[1],
            gdal.GCI_Undefined,
        )
        band.SetColorInterpretation(color_interpretation)
        # Here we could set band metadata
        # See:
        # https://gdal.org/en/stable/drivers/raster/gtiff.html#metadata
        # https://gdal.org/en/stable/user/raster_data_model.html#imagery-domain-remote-sensing
    # Close dataset
    ds = None


def main() -> None:
    """Build multiband Image VRT."""
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

    # Get input directory
    input_dir = Path(options["input"])

    # Get output file path for VRT
    output = Path(options["output"])

    data_type = None
    if options["data_type"]:
        data_type = options["data_type"]

    resolution = None
    if options["resolution"]:
        resolution = options["resolution"]
        if resolution.isdigit():
            resolution = int(resolution)
        elif resolution not in {"highest", "lowest", "average"}:
            gs.fatal(
                _(
                    'Invalid resolution value: "{resolution}". '
                    'Must be an integer, "highest", "lowest", or "average".',
                ).format(resolution=resolution),
            )

    build_vrt(
        input_dir,
        output,
        band_config,
        data_type=data_type,
        resolution=resolution,
        multiband=not flags["s"],
        raster_file_pattern=options["file_pattern"],
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
