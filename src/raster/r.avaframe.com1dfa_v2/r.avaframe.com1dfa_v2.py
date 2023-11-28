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
# % key: id
# % type: string
# % required: yes
# % multiple: no
# % description: Id of release area
# %end

# %option
# % key: entrainment_area
# % type: string
# % required: no
# % multiple: no
# % options: yes,no
# % answer: no
# % description: Use entrainment area in model
# %end

# %option
# % key: resistance_area
# % type: string
# % required: no
# % multiple: no
# % options: yes,no
# % answer: no
# % description: Use resistance area in model
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

# %option G_OPT_M_DIR
# % key: export_directory
# % required: no
# % multiple: no
# % description: Directory where resulting raster maps should be stored as GeoTiff
# %end

# %option
# % key: ppr
# % type: string
# % required: no
# % multiple: no
# % description: Name of the output imagery group for ppr
# %end

# %option
# % key: pft
# % type: string
# % required: no
# % multiple: no
# % description: Name of the output imagery group for pft
# %end

# %option
# % key: pfv
# % type: string
# % required: no
# % multiple: no
# % description: Name of the output imagery group for pfv
# %end

# %option
# % key: format
# % type: string
# % required: no
# % multiple: no
# % options: csv, json
# % description: Format for output of simulation report to stdout
# %end

# %option G_OPT_M_NPROCS
# %end

# %rules
# % excludes: export_directory,pft,pfv,ppr
# % required: export_directory,pft,pfv,ppr
# % collective: pft,pfv,ppr
# %end

import os
import sys

from functools import partial
from pathlib import Path
from multiprocessing import Pool
from urllib import parse

# Local imports
import grass.script as gs


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
    return 0


def link_result(asc_path):
    """Link output ASCII to mapset"""
    Module(
        "r.external",
        flags="or",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )
    return 0


def import_result(asc_path):
    """Import output ASCII into mapset"""
    Module(
        "r.in.gdal",
        flags="o",
        input=str(asc_path),
        output=asc_path.stem,
        quiet=True,
    )
    return 0


def convert_result(
    asc_path, config=None, results_df=None, format="GTiff", directory="/tmp"
):
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
            str(
                [
                    int(results_df.loc[idx] * 100)
                    for idx in results_df.index
                    if idx in mapname
                ][0]
            ),
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
    avalanche_dir = avalanche_base_dir / gs.tempname(12)
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
        3: "Wetsnow",
    }

    buffer = float(options["buffer"])

    if options["export_directory"]:
        if not Path(options["export_directory"]).exists():
            gs.fatal(
                _("Directory <{}> does not exist".format(options["export_directory"]))
            )
        if not os.access(options["export_directory"], os.W_OK):
            gs.fatal(
                _("Directory <{}> is not writable".format(options["export_directory"]))
            )

    # Get release area
    release_area = "https://gis3.nve.no/arcgis/rest/services/featureservice/AlarmInput/FeatureServer/0/query?where=id+%3D+{id}&outFields=*&f=json".format(id = options["id"])
    ogr_dataset_release_area = gdal.OpenEx(release_area, gdal.OF_VECTOR)

    # actinia requires input URLs to be quoted if eg & is used
    if not ogr_dataset_release_area:
        ogr_dataset_release_area = gdal.OpenEx(
            parse.unquote(release_area), gdal.OF_VECTOR
        )
    layer_release_area = ogr_dataset_release_area.GetLayerByIndex(0)
    release_extent = layer_release_area.GetExtent()  # Extent is west, east, south, north
    config = dict(layer.GetNextFeature())  # first feature contains config attributes

    # Get entrainment area
    if options["entrainment_area"] == "yes":
        entrainment_area = "https://gis3.nve.no/arcgis/rest/services/featureservice/AlarmInput/FeatureServer/1/query?where=id+%3D+{id}&outFields=*&f=json".format(id = options["id"])
        ogr_dataset_entrainment_area = gdal.OpenEx(entrainment_area, gdal.OF_VECTOR)
        # actinia requires input URLs to be quoted if eg & is used
        if not ogr_dataset_entrainment_area:
            ogr_dataset_entrainment_area = gdal.OpenEx(
                parse.unquote(entrainment_area), gdal.OF_VECTOR
            )
        layer_entrainment_area = ogr_dataset_entrainment_area.GetLayerByIndex(0)
        config_entrainment_area = dict(layer_entrainment_area.GetNextFeature())  # first feature contains config attributes
        entries_to_remove = ('OBJECTID', 'Id', 'Shape__Area', 'Shape__Length')
        for key in entries_to_remove:
            if key in config_entrainment_area:
                del config_entrainment_area[key]
        config.update(config_entrainment_area)


    # Get resistance area
    if options["resistance_area"] == "yes":
    entrainment_area = "https://gis3.nve.no/arcgis/rest/services/featureservice/AlarmInput/FeatureServer/2/query?where=id+%3D+{id}&outFields=*&f=json".format(id = options["id"])
    ogr_dataset_resistance_area = gdal.OpenEx(resistance_area, gdal.OF_VECTOR)
    # actinia requires input URLs to be quoted if eg & is used
        if not ogr_dataset_resistance_area:
            ogr_dataset_resistance_area = gdal.OpenEx(
                parse.unquote(resistance_area), gdal.OF_VECTOR
            )
        layer_resistance_area = ogr_dataset_resistance_area.GetLayerByIndex(0)
        config_resistance_area = dict(layer_resistance_area.GetNextFeature())  # first feature contains config attributes
        entries_to_remove = ('OBJECTID', 'Id', 'Shape__Area', 'Shape__Length')
        for key in entries_to_remove:
            if key in config_resistance_area:
                del config_resistance_area[key]
        config.update(config_resistance_area)

    # Currently hardcoded settings
    if config["multipleRelTh_m"]:
        release_thicknesses = [
            for snow_depth in config["multipleRelTh_m"].split(",")
        ]
    else:
        release_thicknesses = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    release_name = f"com1DFA_v2_{config['id']}"

    # Define directory for simulations
    avalanche_dir = Path(gs.tempfile(create=False))

    # Create simulation base directory
    (avalanche_dir).mkdir(mode=0o777, parents=True, exist_ok=True)

    # Set relevant region from release area, buffer and DTM
    region = gs.parse_command(
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
    config["frictionModel"] = friction_model_dict[config["frictionModel"]]

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
        str(avalanche_dir / f"{release_name}_release_area.shp"),
        ogr_dataset_release_area,
        options='-f "ESRI Shapefile"',
    )

    gdal.VectorTranslate(
        str(avalanche_dir / f"{release_name}_entrainment_area.shp"),
        ogr_dataset_entrainment_area,
        options='-f "ESRI Shapefile"',
    )

    gdal.VectorTranslate(
        str(avalanche_dir / f"{release_name}_resistance_area.shp"),
        ogr_dataset_resistance_area,
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

    com1dfa_results_pd = pd.concat(
        [com1dfa_results[3] for com1dfa_results in com1dfa_results_list]
    )

    if options["format"]:

        if options["format"] == "json":
            print(com1dfa_results_pd.to_json())
        if options["format"] == "csv":
            print(com1dfa_results_pd.to_csv())

    # Link or import result ASCII files
    result_files = list((avalanche_dir).rglob("**/Outputs/com1DFA/peakFiles/*.asc"))
    with Pool(min(int(options["nprocs"]), len(result_files))) as pool:
        if options["export_directory"]:
            convert_result_gtiff = partial(
                convert_result,
                results_df=com1dfa_results_pd["relTh"],
                config=config,
                format="GTiff",
                directory=options["export_directory"],
            )
            pool.map(convert_result_gtiff, result_files)
        elif flags["l"]:
            pool.map(link_result, result_files)
        else:
            pool.map(import_result, result_files)

    # Create imagery group from results
    if not options["export_directory"]:
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
    options, flags = gs.parser()

    # lazy imports
    from grass.pygrass.modules.interface import Module

    try:
        from avaframe.in3Utils import initializeProject
        from avaframe.com1DFA import com1DFA
        from avaframe.in3Utils import logUtils
        from avaframe.in3Utils import cfgUtils
    except ImportError:
        gs.fatal(_("Unable to load avaframe library"))
    try:
        from osgeo import gdal
    except ImportError:
        gs.fatal(_("Unable to load GDAL library"))
    try:
        import pandas as pd
    except ImportError:
        gs.fatal(_("Unable to load pandas library"))

    sys.exit(main())
