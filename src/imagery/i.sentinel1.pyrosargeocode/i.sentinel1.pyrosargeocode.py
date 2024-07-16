#!/usr/bin/env python3

"""
MODULE:       i.sentinel1.pyrosargeocode
AUTHOR(S):    Stefan Blumentrath
PURPOSE:      Pre-process and import Sentinel-1 imagery using pyroSAR / ESA SNAP
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
- Support SLC workflows
- address pyrosar issues:
  - return resulting file names and not just processing XML
  - unclear parallelization
  - handling of axis order / GeoJSON in Spatialist
  - ...
"""

# %module
# % description: Pre-process and import Sentinel-1 imagery using pyroSAR / ESA SNAP
# % keyword: import
# % keyword: raster
# % keyword: imagery
# % keyword: copernicus
# % keyword: sentinel
# % keyword: satellite
# % keyword: radar
# %end

# %option G_OPT_F_INPUT
# %key: input
# % description: Comma separated list of paths to Sentinel-1 SAFE files or single text file with list of Sentinel-1 SAFE files (one path per row)
# % label: Input file or directory with Sentinel-1 imagery
# %end

# %option G_OPT_M_DIR
# % key: output_directory
# % required: no
# % description: Name for output directory where geocoded Sentinel-1 data is stored (default: ./)
# % label: Directory where geocoded Sentinel-1 data is stored
# %end

# %option G_OPT_F_INPUT
# %key: aoi
# % required: no
# % description: Path to GeoJSON file with the Area Of Interest (aoi)
# % label: Path to GeoJSON file with the Area Of Interest (aoi)
# %end

# %option
# %key: elevation
# % type: string
# % required: yes
# % multiple: no
# % description: Digital elevation model to use for geocoding (either a path to a GeoTiff or a linked raster map)
# %end

# %option
# %key: auxillary_data
# % type: string
# % required: no
# % multiple: yes
# % options: incidenceAngleFromEllipsoid,localIncidenceAngle,projectedLocalIncidenceAngle,DEM,layoverShadowMask,scatteringArea,gammaSigmaRatio
# % answer: incidenceAngleFromEllipsoid,localIncidenceAngle,projectedLocalIncidenceAngle,layoverShadowMask
# % description: Auxillary data to include in the output (scatteringArea and gammaSigmaRation require the n-flag)
# %end

# %option
# % key: polarizations
# % type: string
# % required: yes
# % multiple: yes
# % options: VV,VH
# % description: Polarizations to process
# %end

# %option
# % key: speckle_filter
# % type: string
# % required: no
# % multiple: no
# % options: boxcar,median,frost,gamma_map,lee,refined_lee,lee_sigma,IDAN,mean
# % description: Apply speckle filter algorithms from ESA SNAP
# %end

# %option G_OPT_M_NPROCS
# %end

# %option G_OPT_M_DIR
# % key: temporary_directory
# % required: no
# % description: Path to the directory where temporary data is stored (default: systems temporary directory)
# % label: Performance can benefit from putting temporary data on a fast storage area
# %end

# %option G_OPT_F_OUTPUT
# %key: register_file
# % required: no
# % description: File to be used to register results in a Space Time Raster Dataset
# %end

# %flag
# % key: s
# % description: Execute each node in graph seperately (can circumvent memory limits but takes more time)
# %end

# %flag
# % key: f
# % description: Fetch precise orbit files if possible
# %end

# %flag
# % key: n
# % description: Apply radiometric normalization
# %end

# %flag
# % key: e
# % description: Input elevation model represents ellipsoidal heights
# % description: If the input elevation model represents ellipsoidal heights, no Earth Gravitational Model is applied during geocoding
# %end

# %flag
# % key: d
# % description: Rescale backscatter to dB
# %end

# %flag
# % key: l
# % description: Link resulting data
# %end

# %flag
# % key: m
# % description: Link resulting data and read statistics from metadata
# %end

# %flag
# % key: r
# % description: Link resulting data and do not read statistics
# %end

import os
import shutil
import sys
import tempfile

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE

import grass.script as gs


def get_raster_gdalpath(map_name, check_linked=True):
    """Get get the path to a raster map that can be opened by GDAL
    Checks for GDAL source of linked raster data and returns those
    if not otherwise requested"""
    if check_linked:
        gis_env = gs.gisenv()
        map_info = gs.find_file(map_name)
        header_path = (
            Path(gis_env["GISDBASE"])
            / gis_env["LOCATION_NAME"]
            / map_info["mapset"]
            / "cell_misc"
            / map_info["name"]
            / "gdal"
        )
        if header_path.exists():
            gdal_path = Path(
                gs.parse_key_val(header_path.read_text().replace(": ", "="))["file"]
            )
            if gdal_path.exists():
                return gdal_path
        gdal_path = Path(
            gs.parse_key_val(
                gs.parse_command("r.info", flags="e", map=map_name)["comments"]
                .replace("\\", "")
                .replace('"', ""),
                vsep=" ",
            )["input"]
        )
        if gdal_path.exists():
            return gdal_path
    gdal_path = Path(gs.find_file(map_name)["file"].replace("/cell/", "/cellhd/"))
    if gdal_path.exists():
        return gdal_path
    gs.fatal(_("Cannot determine GDAL readable path to raster map {}").format(map_name))


def get_aoi_geometry(geojson_file):
    """Extract the Area of Interest AOI from a GeoJSON file and
    return it as an OGR Geometry object.
    The input GeoJSON should contain only one polygon geometry"""

    ogr_dataset = ogr.Open(
        f"/vsicurl/{geojson_file}" if geojson_file.startswith("http") else geojson_file
    )
    # Check OGR dataset content
    if not ogr_dataset:
        gs.fatal(_("Could not open AOI file <{}>").format(geojson_file))
    if ogr_dataset.GetLayerCount() > 1:
        gs.warning(
            _("Input file contains more than one layer. Using only the first one.")
        )
    ogr_layer = ogr_dataset.GetLayerByIndex(0)
    if ogr_layer.GetGeomType() != 3:
        gs.fatal(_("GeoJSON does not contain polygons"))
    if ogr_layer.GetFeatureCount() > 1:
        gs.warning(
            _("GeoJSON contains more than one geometry. Using only the first one.")
        )
    ogr_feature = ogr_layer.GetFeature(0)

    return ogr_feature.geometry().ExportToWkt()


def get_target_geometry(s1_file, aoi=None):
    """Intersect the scene geometry of a pyrosar scene identification
    with an area of interest WKT polygon in EPSG:4326 and returns:
    - True if the scene footprint is located fully within the aoi geometry
    - False if the scene footprint and the aoi geometry are disjoined
    - path to a GeoJSON of the intersection geometry if aoi and scene
      footprint intersect
    """

    s_srs = osr.SpatialReference()
    s_srs.ImportFromEPSG(4326)

    # spaialist Vector in pyrosar expects lon-lat order (CRS84)
    t_srs = osr.SpatialReference()
    t_srs.ImportFromEPSG(4326)
    t_srs.SetDataAxisToSRSAxisMapping([2, 1])

    # JSON export from spaialist is not valid GeoJSON
    bpol_geom = ogr.CreateGeometryFromWkt(s1_file.geometry().convert2wkt()[0], s_srs)
    bpol_geom.FlattenTo2D()

    aoi = ogr.CreateGeometryFromWkt(aoi, s_srs)

    if bpol_geom.Within(aoi):
        return True
    if not bpol_geom.Intersect(aoi):
        gs.warning(
            _("Sentinel-1 scene <{scene}> does not intersect with given AOI").format(
                scene=Path(s1_file.scene).stem
            )
        )
        return False
    bpol = bpol_geom.Intersection(aoi)
    bpol.AssignSpatialReference(t_srs)
    aoi = Path(f"{gs.tempfile(create=False)}.geojson")
    Path(aoi).write_text(bpol.ExportToJson(), encoding="UTF8")
    return str(aoi)


def get_gpt_options(nprocs):
    """Get options for GPT execution"""
    try:
        gpt = shutil.which("gpt")
        gpt_vmoptions = Path(f"{gpt}.vmoptions").read_text(encoding="UTF8").split("\n")
        xmx_option = [opt for opt in gpt_vmoptions if opt.startswith("-Xmx")][
            0
        ].replace("-Xmx", "")
    except RuntimeError:
        gs.fatal(_(""))
    return gpt, [
        "-x",
        "-c",
        f"{int(float(xmx_option[:-1])*0.5)}{xmx_option[-1]}",
        "-q",
        str(nprocs),
    ]


def process_image_file(
    s1_file,
    kwargs=None,
    import_flags=None,
    aoi=None,
):
    """Preprocess and import SAFE file
    return string to register map in TGIS"""

    s1_file_id = identify(str(s1_file))

    if s1_file_id.is_processed(kwargs["outdir"]):
        gs.warning(_("Scene {} is already processed").format(s1_file.name))
        if not gs.overwrite():
            return None
        # Remove results if overwrite is requested
        gs.warning(_("Removing results from prior processes as overwrite is requested"))
        for existing_file in Path(kwargs["outdir"]).glob(
            f"{s1_file_id.outname_base()}*"
        ):
            existing_file.unlink()

    # Get S1 metadata and set scene specific kwargs
    start_time = (
        s1_file_id.meta["acquisition_time"]["start"].replace("T", " ").split(".")[0]
    )
    end_time = (
        s1_file_id.meta["acquisition_time"]["stop"].replace("T", " ").split(".")[0]
    )

    if import_flags["f"]:
        osvdir = Path().home() / ".snap" / "auxdata" / "Orbits" / "Sentinel-1"
        osvdir.mkdir(exist_ok=True, parents=True)

        if not s1_file_id.getOSV(osvdir=osvdir, osvType="POE", returnMatch=True):
            gs.warning(
                _("Could not fetch Precise Orbit file for scene {}").format(
                    s1_file.name
                )
            )
            kwargs["allow_RES_OSV"] = True

    if aoi:
        aoi = get_target_geometry(s1_file_id, aoi=aoi)
        if not aoi:
            return None
        if not isinstance(aoi, bool):
            kwargs["shapefile"] = aoi

    # Apply processing graph
    gs.verbose(_("Start geocoding scene {}").format(s1_file.name))
    try:
        snap.geocode(infile=s1_file_id, **kwargs)
    except RuntimeError as runtime_error:
        gs.fatal(
            _(
                "Geocoding failed with the following error: {}\nPlease check the log files"
            ).format(runtime_error)
        )

    register_strings = []
    output_dir = Path(kwargs["outdir"])
    orbit_direction = "ascending" if s1_file_id.orbit == "A" else "descending"
    polarizations = kwargs["polarizations"]
    polarizations.extend(kwargs["export_extra"])
    for polarization in polarizations:
        if polarization in kwargs["export_extra"]:
            semantic_label = (
                f"S1_{orbit_direction}_{s1_file_id.orbitNumber_rel}_{polarization}"
            )
            output_tif = output_dir / f"{s1_file_id.outname_base()}_{polarization}.tif"
        else:
            geocoding = "rtc" if import_flags["n"] else "elp"
            scale = "_db" if import_flags["d"] else ""
            semantic_label = f"S1_{orbit_direction}_{s1_file_id.orbitNumber_rel}_{polarization}_{kwargs['refarea']}_{geocoding}{scale}"
            # e.g.: S1A__IW___D_20230602T043121_VV_gamma0-elp_db.tif
            output_tif = (
                output_dir
                / f"{s1_file_id.outname_base()}_{polarization}_{kwargs['refarea']}-{geocoding}{scale}.tif"
            )

        output_map = gs.utils.legalize_vector_name(output_tif.stem)

        # Import resulting geotiff
        import_kwargs = {
            "memory": 2048,
            "input": str(output_tif),
            "band": 1,
            "output": output_map,
        }
        module = "r.in.gdal"

        if any([import_flags["l"], import_flags["m"], import_flags["r"]]):
            module = "r.external"
            import_kwargs["flags"] = "m" if import_flags["m"] else ""
            import_kwargs["flags"] = "r" if import_flags["r"] else ""
            import_kwargs.pop("memory")

        gs.verbose(_("Importing file {}").format(str(output_tif)))
        Module(module, stderr_=PIPE, **import_kwargs)
        Module(
            "r.support",
            stderr_=PIPE,
            map=output_map,
            semantic_label=semantic_label,
        )
        register_strings.append(
            f"{output_map}|{start_time}|{end_time}|{semantic_label}"
        )
    return "\n".join(register_strings)


def check_directory(directory, write_mode=True):
    """Check if directory exists and has required access rights"""
    mode = os.W_OK if write_mode else os.R_OK
    directory = Path(directory)
    if directory.exists():
        if os.access(str(directory), mode):
            return directory
        gs.fatal(
            _("Directory <{dir}> is not {dir_mode}").format(
                dir=str(directory),
                dir_mode="writable" if write_mode else "readable",
            )
        )
    else:
        try:
            directory.mkdir(exist_ok=True, parents=True)
        except OSError:
            gs.fatal(
                _("Directory <{dir}> does not exist and cannot be created").format(
                    str(directory)
                )
            )
    return directory


def check_files_list(file_path_list):
    """Checks files in a list of files exist and gives a warning otherwise"""
    existing_paths = []
    for file_path in file_path_list:
        file_path_object = Path(file_path)
        if file_path_object.exists():
            existing_paths.append(file_path_object)
        else:
            gs.warning(_("File {} not found").format(file_path))

    return existing_paths


def main():
    """Do the main work"""

    # Check if gpt is available
    if not shutil.which("gpt"):
        gs.fatal(_("gpt commandline tool from ESA SNAP not found on current PATH"))

    export_extra = None
    if options["auxillary_data"]:
        export_extra = options["auxillary_data"].split(",")
        if "scatteringArea" in export_extra and not flags["n"]:
            gs.fatal(_("scatteringArea requires terrain flattening (n-flag)"))
        if "gammaSigmaRatio" in export_extra and not flags["n"]:
            gs.fatal(_("gammaSigmaRatio requires terrain flattening (n-flag)"))

    # Check if GDAL-GRASS driver is available
    if "GRASS" not in [
        gdal.GetDriver(idx).GetDescription() for idx in range(gdal.GetDriverCount())
    ]:
        gs.fatal(_("GDAL-GRASS driver not available"))

    # Identify and check S1 files to geocode
    file_input = options["input"].split(",")
    if len(file_input) == 1:
        file_input = Path(file_input[0])
        if file_input.is_dir() and file_input.suffix.upper() != ".SAFE":
            # Directory mode
            file_input = list(file_input.glob("S1*.SAFE"))
        elif (
            file_input.suffix.upper() == ".SAFE" or file_input.suffix.lower() == ".zip"
        ):
            # Single file mode
            file_input = [file_input]
        else:
            # Text file mode
            file_input = file_input.read_text(encoding="UTF8").split("\n")

    file_input = check_files_list(file_input)

    if len(file_input) < 1:
        gs.fatal(_("No Sentinel-1 files found to Geocode"))

    # Get target CRS
    location_crs_wkt = gs.read_command("g.proj", flags="w").strip()

    # Get info on target DEM
    dem_info = gs.raster_info(options["elevation"])
    if dem_info["nsres"] != dem_info["ewres"]:
        gs.warning(
            _(
                "Cells NS- and EW-resolution in raster map <{}> differ. Using NS-resolution."
            ).format(options["elevation"])
        )

    dem_info["GDAL_path"] = get_raster_gdalpath(options["elevation"])

    nprocs = int(options["nprocs"])
    nprocs_outer = min(nprocs, len(file_input))
    nprocs_inner = nprocs if len(file_input) == 1 else 1

    speckle_filter_dict = {
        "boxcar": "Boxcar",
        "median": "Median",
        "frost": "Frost",
        "gamma_map": "Gamma Map",
        "lee": "Lee",
        "refined_lee": "Refined Lee",
        "lee_sigma": "Lee Sigma",
        "IDAN": "IDAN",
        "mean": "Mean",
    }

    # Check output directory
    output_directory = check_directory(options["output_directory"])
    if options["temporary_directory"]:
        check_directory(options["temporary_directory"])

    # Check if gpt is available
    gpt_options = get_gpt_options(nprocs_inner)

    # Setup function for geocoding
    geocode_kwargs = {
        "t_srs": location_crs_wkt,
        "returnWF": True,
        "terrainFlattening": flags["n"],
        "refarea": "gamma0" if flags["n"] else "sigma0",
        "export_extra": export_extra,
        "outdir": str(output_directory),
        "speckleFilter": (
            speckle_filter_dict[options["speckle_filter"]]
            if options["speckle_filter"]
            else None
        ),
        "spacing": float(dem_info["nsres"]),
        "externalDEMFile": str(dem_info["GDAL_path"]),
        "externalDEMNoDataValue": (
            -2147483678.0 if dem_info["datatype"] == "DCELL" else None
        ),
        "externalDEMApplyEGM": not flags["e"],
        "alignToStandardGrid": True,
        "demResamplingMethod": "BILINEAR_INTERPOLATION",
        "imgResamplingMethod": "BILINEAR_INTERPOLATION",
        "standardGridOriginX": float(dem_info["west"]),
        "standardGridOriginY": float(dem_info["north"]),
        "clean_edges": True,
        "polarizations": options["polarizations"].split(","),
        "scaling": "dB" if flags["d"] else "linear",
        "groupsize": 1 if flags["s"] else 999,  # 1 = execute each node separately
        "gpt_args": gpt_options[1],
        "tmpdir": options["temporary_directory"] or tempfile.gettempdir(),
    }

    if export_extra and "gammaSigmaRatio" in export_extra:
        geocode_kwargs["refarea"] = ["sigma0", "gamma0"]

    # Pre-configure geocode function
    _geocode_snap = partial(
        process_image_file,
        kwargs=geocode_kwargs,
        aoi=get_aoi_geometry(options["aoi"]) if options["aoi"] else None,
        import_flags=flags,
    )

    # Execute geocoding (in paralell if requested)
    if nprocs_outer > 1:
        with Pool(nprocs_outer) as pool:
            geocoded_files = pool.map(_geocode_snap, file_input)
    else:
        geocoded_files = []
        for s1_file in file_input:
            geocoded_files.append(_geocode_snap(s1_file))

    geocoded_files = [
        geocoded_file for geocoded_file in geocoded_files if geocoded_file
    ]

    if geocoded_files:
        if options["register_file"]:
            # Write registration files
            Path(options["register_file"]).write_text(
                "\n".join(geocoded_files), encoding="UTF8"
            )
        else:
            print("\n".join(geocoded_files))


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    from grass.pygrass.modules.interface import Module

    try:
        from osgeo import gdal, ogr, osr
    except ImportError:
        gs.fatal(
            _(
                "Can not import GDAL python bindings. Please install it with 'pip install GDAL==${GDAL_VERSION}'"
            )
        )

    try:
        from pyroSAR import identify
        from pyroSAR import snap
    except ImportError:
        gs.fatal(
            _(
                "Can not import pyroSAR library. Please install it with 'pip install pyrosar'"
            )
        )

    sys.exit(main())
