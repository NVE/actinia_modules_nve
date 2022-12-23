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

# Get ID, rho, rhoEnt, shape
from pathlib import Path
from multiprocessing import Pool

# Local imports
import grass.script as gscript


def write_avaframe_config(
    config_file_path,
    rho=None,
    rho_ent=None,
    mesh_cell_size=None,
    friction_model="samosAT",
    release_thickness="0.5",
    release_thickness_range_variation="+3.5$8",
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
            elif line.startswith("relThFromShp ="):
                line = "relThFromShp = False"
            elif line.startswith("relTh ="):
                line = f"relTh = {release_thickness}"
            elif line.startswith("relThRangeVariation ="):
                line = f"relThRangeVariation = {release_thickness_range_variation}"

            cfg_file.write(line + "\n")


def link_result(asc_path):
    # Export DTM to ASCII
    Module(
        "r.external",
        flags="or",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )


def import_result(asc_path):
    # Export DTM to ASCII
    Module(
        "r.in.gdal",
        flags="o",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )


def main():
    """Run com1DFA simulation from Avaframe with selected configuration"""
    # options = {"elevation": "DTM_10m@DTM",
    #     "release_area": "https://gis3.nve.no/arcgis/rest/services/"
    #         "featureservice/AlarmSamosAT/MapServer/"
    #         "0/query?where=objectid+%3D+1&outFields=*&f=json",
    #     "buffer": 1000,
    # }

    buffer = float(options["buffer"])

    friction_model_dict = {
        1: "samosAT",
        2: "Coulomb",
        3: "Voellmy",
    }

    release_name = "NonsnibbaRelease"

    # Get release area
    ogr_dataset = gdal.OpenEx(options["release_area"], gdal.OF_VECTOR)
    layer = ogr_dataset.GetLayerByIndex(0)
    release_extent = layer.GetExtent()  # Extent is west, east, south, north
    config = layer.GetFeature(1)

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

    # Define directory for simulations
    avalanche_dir = Path(gscript.tempfile(create=False))

    # Configue avaframe
    config_main = cfgUtils.getGeneralConfig()
    config_main["FLAGS"]["savePlot"] = "False"
    config_main["FLAGS"]["ReportDir"] = "False"
    config_main["FLAGS"]["reportOneFile"] = "False"
    config_main["MAIN"]["avalancheDir"] = str(avalanche_dir)

    # Initialize project
    initializeProject.initializeFolderStruct(config_main["MAIN"]["avalancheDir"])

    # Write release area to shape
    gdal.VectorTranslate(
        str(avalanche_dir / "Inputs" / "REL" / f"{release_name}.shp"),
        ogr_dataset,
        options='-f "ESRI Shapefile"',
    )

    # Export DTM to ASCII
    Module(
        "r.out.gdal",
        input=options["elevation"],
        output=str(avalanche_dir / "Inputs" / "raster.asc"),
        format="AAIGrid",
        overwrite=True,
        verbose=True,
    )

    # Create ini-file with configuration
    cfg_ini_file = avalanche_dir / "cfg.ini"
    write_avaframe_config(
        cfg_ini_file,
        # density of snow [kg/m³]
        rho=config["rho_kgPerSqM"],
        # density of entrained snow [kg/m³]
        rho_ent=config["rhoEnt_kgPerSqM"],
        # friction model (samosAT, Coulomb, Voellmy)
        friction_model=friction_model_dict[config["frictModel"]],
        mesh_cell_size=region["nsres"],
        release_thickness=float(config["snowDepth_cm"]) / 100.0,
        release_thickness_range_variation="+3.5$8",
    )

    # Start logging
    log = logUtils.initiateLogger(str(avalanche_dir), "r.avaframe")
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", str(avalanche_dir))

    # call com1DFA and perform simulations
    com1dfa_results = com1DFA.com1DFAMain(
        str(avalanche_dir), config_main, cfgFile=cfg_ini_file
    )

    if options["format"] == "json":
        print(com1dfa_results[3].to_json())
    if options["format"] == "csv":
        print(com1dfa_results[3].to_csv())

    # Link result ASCII files
    result_files = list(
        (avalanche_dir / "Outputs" / "com1DFA" / "peakFiles").glob("*.asc")
    )
    with Pool(int(options["nprocs"])) as pool:
        if flags["l"]:
            pool.map(link_result, result_files)
        else:
            pool.map(import_result, result_files)

    # Create imagery group from results
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
        # Module(
        #     "r.out.gdal",
        #     flags="cmf",
        #     input=result_type,
        #     output=str(avalanche_dir / f"{result_type}.tif"),
        #     format="COG",
        #     type="Float32",
        #     createopt="PREDICTOR=FLOATING_POINT",
        #     overwrite=True,
        # )


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
