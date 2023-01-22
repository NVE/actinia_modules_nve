#!/usr/bin/env python3

"""
 MODULE:       r.avaframe.com1dfa
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Run com1dfa avalanche simulations using AvaFrame
 COPYRIGHT:    (C) 2022 by Stefan Blumentrath

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

"""

# %module
# % description: Run com1dfa avalanche simulations using AvaFrame
# % keyword: avalanche
# % keyword: simulation
# % keyword: avaframe
# % keyword: com1dfa
# % keyword: raster
# % keyword: terrain
# % keyword: dtm
# %end

# %flag
# % key: l
# % description: Link temporary results (do not import)
# %end

# %option
# % key: release_area
# % type: string
# % required: yes
# % multiple: no
# % description: Path or URL to OGR readable vector dataset with release area(s)
# %end

# %option G_OPT_R_ELEV
# %end

# %option
# % key: buffer
# % type: integer
# % multiple: no
# % answer: 2000
# % description: Buffer around release area for area of interest
# %end

# %option
# % key: ppr
# % type: string
# % required: yes
# % multiple: no
# % description: Name of the output imagery group for ppr
# %end

# %option
# % key: pft
# % type: string
# % required: yes
# % multiple: no
# % description: Name of the output imagery group for pft
# %end

# %option
# % key: pfv
# % type: string
# % required: yes
# % multiple: no
# % description: Name of the output imagery group for pfv
# %end

# %option
# % key: format
# % type: string
# % required: no
# % multiple: no
# % options: csv, json
# % answer: csv
# % description: Format for output of simulation report to stdout
# %end

# %option G_OPT_M_NPROCS
# %end

import sys

from functools import partial
from pathlib import Path
from multiprocessing import Pool
from urllib import parse

# Local imports
import grass.script as gscript


def write_avaframe_config(
    config_file_path,
    rho=None,
    rho_ent=None,
    mesh_cell_size=None,
    friction_model="samosAT",
    release_thickness="0.5",
    # release_thickness_range_variation="+3.5$8",
):
    """Write Avaframe config file"""
    with open(config_file_path, "w", encoding="utf8") as cfg_file:
        # Get default config
        for line in (
            (Path(com1DFA.__file__).parent / "com1DFACfg.ini")
            .read_text(encoding="utf8")
            .split("\n")
        ):
            # Replace with given values
            if line.startswith("rho =") and rho:
                line = f"rho = {rho}"
            elif line.startswith("rhoEnt =") and rho_ent:
                line = f"rhoEnt = {rho_ent}"
            elif line.startswith("frictModel =") and friction_model:
                line = f"frictModel = {friction_model}"
            elif line.startswith("meshCellSize =") and mesh_cell_size:
                line = f"meshCellSize = {mesh_cell_size}"
            elif line.startswith("sphKernelRadius =") and mesh_cell_size:
                line = f"sphKernelRadius = {mesh_cell_size}"
            elif line.startswith("relThFromShp ="):
                line = "relThFromShp = False"
            elif line.startswith("relTh ="):
                line = f"relTh = {release_thickness}"
            # elif line.startswith("relThRangeVariation ="):
            #     line = f"relThRangeVariation = {release_thickness_range_variation}"

            cfg_file.write(line + "\n")


def link_result(asc_path):
    """Link output ASCII to mapset"""
    Module(
        "r.external",
        flags="or",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )


def import_result(asc_path):
    """Import output ASCII into mapset"""
    Module(
        "r.in.gdal",
        flags="o",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )


def convert_result(asc_path, config=None, format="GTiff", directory="/tmp"):
    """Convert ascii file to GeoTiff"""
    result_prefix = {
        "pft": "MaxFlowThickness",
        "pfv": "MaxFlowVelocity",
        "ppr": "MaxPressure",
    }
    mapname = asc_path.stem
    # {Produktnavn}_{objectid}_{snowDepth_cm}_{rho_kgPerSqM}_{rhoEnt_kgPerSqM}_{frictModel}.tif
    gtiff_name = "_".join(
        [
            result_prefix[mapname[-3:]],
            str(config["OBJECTID"]),
            str(config["snowDepth_cm"]),
            str(config["rho_kgPerSqM"]),
            str(config["rhoEnt_kgPerSqM"]),
            str(config["frictModel"]),
        ]
    )
    Module(
        "r.external",
        flags="or",
        input=str(asc_path),
        output=mapname,
        quiet=True,
    )
    Module(
        "r.out.gdal",
        flags="f",
        input=mapname,
        output=str(Path(directory) / f"{gtiff_name}.tif"),
        format="GTiff",
        createopt="COMPRESS=LZW,PREDICTOR=3",
        type="Float32",
    )
    return 0


def run_com1dfa(thickness, config_dict=None):
    """Run com1DFA for given thickness"""
    thickness_str = str(thickness).replace(".", ".")

    avalanche_base_dir = config_dict["avalanche_dir"]
    avalanche_dir = avalanche_base_dir / gscript.tempname(12)
    config_dict["main"]["MAIN"]["avalancheDir"] = str(avalanche_dir)

    # Create simulation directory
    (avalanche_dir).mkdir(mode=0o777, parents=True, exist_ok=True)

    # Initialize simulation
    initializeProject.initializeFolderStruct(
        config_dict["main"]["MAIN"]["avalancheDir"]
    )

    # Link input data
    for shape_file in avalanche_base_dir.glob(f"{config_dict['release_name']}*"):
        (avalanche_dir / "Inputs" / "REL" / shape_file.name).symlink_to(shape_file)
    (avalanche_dir / "Inputs" / "raster.asc").symlink_to(
        avalanche_base_dir / "raster.asc"
    )

    # Create ini-file with configuration
    cfg_ini_file = avalanche_dir / f"cfg_{thickness_str}.ini"
    write_avaframe_config(
        cfg_ini_file,
        # density of snow [kg/m³]
        rho=config_dict["rho_kgPerSqM"],
        # density of entrained snow [kg/m³]
        rho_ent=config_dict["rhoEnt_kgPerSqM"],
        # friction model (samosAT, Coulomb, Voellmy)
        friction_model=config_dict["frictModel_name"],
        mesh_cell_size=config_dict["mesh_cell_size"],
        release_thickness=thickness,
        # release_thickness_range_variation="+3.5$8",
    )

    # Start logging
    log = logUtils.initiateLogger(str(avalanche_dir), "r.avaframe.com1dfa")
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", str(avalanche_dir))

    # call com1DFA and perform simulations
    return com1DFA.com1DFAMain(
        str(avalanche_dir), config_dict["main"], cfgFile=cfg_ini_file
    )


def main():
    """Run com1DFA simulation from Avaframe with selected configuration"""
    # options = {"elevation": "DTM_10m@DTM",
    #     "release_area": "https://gis3.nve.no/arcgis/rest/services/"
    #         "featureservice/AlarmSamosAT/MapServer/"
    #         "0/query?where=objectid+%3D+1&outFields=*&f=json",
    #     "buffer": 1000,
    # }
    friction_model_dict = {
        0: "samosAT",
        1: "Coulomb",
        2: "Voellmy",
    }

    buffer = float(options["buffer"])

    # Get release area
    ogr_dataset = gdal.OpenEx(options["release_area"], gdal.OF_VECTOR)

    # actinia requires input URLs to be quoted if eg & is used
    if not ogr_dataset:
        ogr_dataset = gdal.OpenEx(
            parse.unquote(options["release_area"]), gdal.OF_VECTOR
        )
    layer = ogr_dataset.GetLayerByIndex(0)
    release_extent = layer.GetExtent()  # Extent is west, east, south, north
    config = dict(layer.GetNextFeature())  # first feature contains config attributes

    # Currently hardcoded settings
    if config["multipleSnowDepth_cm"]:
        release_thicknesses = list(
            map(float, config["multipleSnowDepth_cm"].split(","))
        )
    else:
        release_thicknesses = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    release_name = f"com1DFA_{config['OBJECTID']}"

    # Define directory for simulations
    avalanche_dir = Path(gscript.tempfile(create=False))

    # Create simulation base directory
    (avalanche_dir).mkdir(mode=0o777, parents=True, exist_ok=True)

    # Set relevant region from release area, buffer and DTM
    region = gscript.parse_command(
        "g.region",
        flags="g",
        align=options["elevation"],
        n=release_extent[3] + buffer,
        s=release_extent[2] - buffer,
        e=release_extent[1] + buffer,
        w=release_extent[0] - buffer,
    )

    config["mesh_cell_size"] = region["nsres"]
    config["avalanche_dir"] = avalanche_dir
    config["frictModel_name"] = friction_model_dict[config["frictModel"]]

    # Configue avaframe
    config_main = cfgUtils.getGeneralConfig()
    config_main["FLAGS"]["savePlot"] = "False"
    config_main["FLAGS"]["createReport"] = "False"
    # config_main["FLAGS"]["reportOneFile"] = "False"
    config_main["MAIN"]["avalancheDir"] = str(avalanche_dir)

    config["main"] = config_main
    config["release_name"] = release_name

    run_com1dfa_thickness = partial(run_com1dfa, config_dict=config)

    # Write release area to shape
    gdal.VectorTranslate(
        str(avalanche_dir / f"{release_name}.shp"),
        ogr_dataset,
        options='-f "ESRI Shapefile"',
    )

    # Export DTM to ASCII
    Module(
        "r.out.gdal",
        input=options["elevation"],
        output=str(avalanche_dir / "raster.asc"),
        nodata=-9999,
        format="AAIGrid",
        overwrite=True,
        verbose=True,
    )

    with Pool(min(int(options["nprocs"]), len(release_thicknesses))) as pool:
        com1dfa_results_list = pool.map(run_com1dfa_thickness, release_thicknesses)

    if options["format"]:
        import pandas as pd

        com1dfa_results_pd = pd.concat(
            [com1dfa_results[3] for com1dfa_results in com1dfa_results_list]
        )
        if options["format"] == "json":
            print(com1dfa_results_pd.to_json())
        if options["format"] == "csv":
            print(com1dfa_results_pd.to_csv())

    # Link or import result ASCII files
    result_files = list((avalanche_dir).rglob("**/Outputs/com1DFA/peakFiles/*.asc"))
    with Pool(min(int(options["nprocs"]), len(result_files))) as pool:
        if flags["l"]:
            pool.map(link_result, result_files)
        if flags["e"]:
            convert_result_gtiff = partial(
                convert_result,
                config=config,
                format="GTiff",
                directory=options["export_directory"],
            )
            pool.map(convert_result_gtiff, result_files)
        else:
            pool.map(import_result, result_files)

    # Create imagery group from results
    if not flags["e"]:
        for result_type in ["ppr", "pft", "pfv"]:
            Module(
                "i.group",
                group=result_type,
                subgroup=result_type,
                input=",".join(
                    [
                        result_file.stem
                        for result_file in result_files
                        if result_file.stem.endswith(result_type)
                    ]
                ),
            )


if __name__ == "__main__":
    options, flags = gscript.parser()

    # lazy imports
    from grass.pygrass.modules.interface import Module

    try:
        from avaframe.in3Utils import initializeProject
        from avaframe.com1DFA import com1DFA
        from avaframe.in3Utils import logUtils
        from avaframe.in3Utils import cfgUtils
    except ImportError:
        gscript.fatal(_("Unable to load avaframe library"))
    try:
        from osgeo import gdal
    except ImportError:
        gscript.fatal(_("Unable to load GDAL library"))

    sys.exit(main())
