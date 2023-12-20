#!/usr/bin/env python3

"""
 MODULE:       r.avaframe.com1dfa
 AUTHOR(S):    Stefan Blumentrath and Yngve Antonsen
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

# %flag
# % key: e
# % description: Use entrainment area in model
# %end

# %flag
# % key: r
# % description: Use resistance area in model
# %end

# %option
# % key: id
# % type: string
# % required: yes
# % multiple: no
# % description: Id of release area
# %end

# %option
# % key: url
# % type: string
# % required: yes
# % multiple: no
# % description: URL to the featureservice of the input data (release, entrainment and resistance area)
# %end

# %option
# % key: release_area_layer_id
# % type: string
# % required: yes
# % multiple: no
# % description: Layer ID for the release area on the feature service.
# %end

# %option
# % key: entrainment_area_layer_id
# % type: string
# % required: no
# % multiple: no
# % description: Layer ID for the entrainment area on the feature service.
# %end

# %option
# % key: resistance_area_layer_id
# % type: string
# % required: no
# % multiple: no
# % description: Layer ID for the resistance area on the feature service.
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
# % requires: resistance_area_layer_id,-r
# % requires: entrainment_area_layer_id,-e
# % requires: -r,resistance_area_layer_id
# % requires: -e,entrainment_area_layer_id
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
    input_config,
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
            #ReleaseArea
            if line.startswith("frictModel =") and input_config["frictionModel"]:
                line = f"frictModel = {input_config['frictionModel']}"
            elif line.startswith("rho =") and input_config["rho_kgPerCubicM"]:
                line = f"rho = {input_config['rho_kgPerCubicM']}"
            elif line.startswith("cpIce =") and input_config["cpIce_joulePerKg"]:
                line = f"cpIce = {input_config['cpIce_joulePerKg']}"
            elif line.startswith("TIni =") and input_config["tIni_degreeCelcius"]:
                line = f"TIni = {input_config['tIni_degreeCelcius']}"
            elif line.startswith("entTempRef =") and input_config["entTemp_degreeCelcius"]:
                line = f"entTempRef = {input_config['entTemp_degreeCelcius']}"
            elif line.startswith("enthRef =") and input_config["enthalpy_joulePerKg"]:
                line = f"enthRef = {input_config['enthalpy_joulePerKg']}"
            #Mangler mu, xsi, tau0, rs0, kappa, r, b
            
            #EntrainmentArea
            elif line.startswith("rhoEnt =") and "rhoEnt_kgPerCubicM" in input_config and input_config["rhoEnt_kgPerCubicM"]:
                line = f"rhoEnt = {input_config['rhoEnt_kgPerCubicM']}"
            elif line.startswith("entEroEnergy =") and "entEro_joulePerKg" in input_config and input_config["entEro_joulePerKg"]:
                line = f"entEroEnergy = {input_config['entEro_joulePerKg']}"
            elif line.startswith("entShearResistance =") and "entShear_joulePerSqM" in input_config and input_config["entShear_joulePerSqM"]:
                line = f"entShearResistance = {input_config['entShear_joulePerSqM']}"
            elif line.startswith("entDefResistance =") and "entDef_joulePerKg" in input_config and input_config["entDef_joulePerKg"]:
                line = f"entDefResistance = {input_config['entDef_joulePerKg']}"
            elif line.startswith("entThFromShp =") and "entTh_m" in input_config and input_config["entTh_m"]:
                line = "entThFromShp = False"
            elif line.startswith("entTh =") and "entTh_m" in input_config and input_config["entTh_m"]:
                line = f"entTh = {input_config['entTh_m']}"

            #Resistance
            elif line.startswith("hRes =") and "hRes_m" in input_config and input_config["hRes_m"]:
                line = f"hRes = {input_config['hRes_m']}"
            elif line.startswith("cw =") and "cw" in input_config and input_config["cw"]:
                line = f"cw = {input_config['cw']}"
            elif line.startswith("dRes =") and "dRes_m" in input_config and input_config["dRes_m"]:
                line = f"dRes = {input_config['dRes_m']}"
            elif line.startswith("sres =") and "sRes_m" in input_config and input_config["sRes_m"]:
                line = f"sres = {input_config['sRes_m']}"
            
            #https://github.com/avaframe/AvaFrame/blob/master/avaframe/com1DFA/com1DFACfg.ini

            elif line.startswith("meshCellSize =") and input_config["mesh_cell_size"]:
                line = f"meshCellSize = {input_config['mesh_cell_size']}"
            elif line.startswith("sphKernelRadius =") and input_config["mesh_cell_size"]:
                line = f"sphKernelRadius = {input_config['mesh_cell_size']}"
            elif line.startswith("relThFromShp ="):
                line = "relThFromShp = False"
            elif line.startswith("relTh ="):
                if input_config['multipleRelTh_m']:
                    line = f"relTh = {input_config['multipleRelTh_m'].replace(',', '|')}"
                else:
                    line = "relTh = ValueError"
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
            str(config["id"]),
            str(
                [
                    results_df.loc[idx]
                    for idx in results_df.index
                    if idx in mapname
                ][0]
            ),
            #str(config["rho_kgPerSqM"]),
            #str(config["relTh"]),
            str(config["frictionModel"]),
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
        flags="cfm",
        input=mapname,
        output=str(Path(directory) / f"{gtiff_name}.tif"),
        format="GTiff",
        createopt="COMPRESS=LZW,PREDICTOR=3",
        type="Float32",
    )
    return 0


def run_com1dfa(config_dict=None):
    """Run com1DFA for given thickness"""

    avalanche_base_dir = config_dict["avalanche_dir"]
    avalanche_dir = avalanche_base_dir / gs.tempname(12)
    config_dict["main"]["MAIN"]["avalancheDir"] = str(avalanche_dir)
    config_dict["main"]["MAIN"]["nCPU"] = config_dict["nCPU"]
    

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
    cfg_ini_file = avalanche_dir / "cfg_avaframe_v2.ini"
    write_avaframe_config(
        cfg_ini_file,
        config_dict,
    )

    # Start logging
    log = logUtils.initiateLogger(str(avalanche_dir), "r.avaframe.com1dfa")
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", str(avalanche_dir))

    # call com1DFA and perform simulations
    return com1DFA.com1DFAMain(
        config_dict["main"], cfgInfo=cfg_ini_file
    )

def get_shape_file_and_config(area_type, module_config, module_options):
    """
    Allowed area_type "RES" and "ENT"
    See avaframe documentation
    """
    area = "{url}/{layer_id}/query?where=id+%3D+{id}&outFields=*&f=json".format(url = module_options["url"], layer_id = module_options[{"ENT": "entrainment_area_layer_id",  "RES": "resistance_area_layer_id"}[area_type]], id = module_options["id"])
    ogr_dataset_area = gdal.OpenEx(area, gdal.OF_VECTOR)
    # actinia requires input URLs to be quoted if eg & is used
    if not ogr_dataset_area:
        ogr_dataset_area = gdal.OpenEx(
            parse.unquote(area), gdal.OF_VECTOR
        )
    layer_area = ogr_dataset_area.GetLayerByIndex(0)
    config_area = dict(layer_area.GetNextFeature())  # first feature contains config attributes
    entries_to_remove = ('OBJECTID', 'Id', 'Shape__Area', 'Shape__Length')
    for key in entries_to_remove:
        if key in config_area:
            del config_area[key]
    module_config.update(config_area)
    (module_config["avalanche_dir"] / area_type).mkdir(parents=True, exist_ok = True)
    gdal.VectorTranslate(
    str(module_config["avalanche_dir"] / area_type / f"{module_config['release_name']}.shp"),
    ogr_dataset_area,
    options='-f "ESRI Shapefile"',
    )

    return module_config

def main():
    """Run com1DFA simulation from Avaframe with selected configuration"""

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
    release_area = "{url}/{layerId}/query?where=id+%3D+{id}&outFields=*&f=json".format(url = options["url"], layerId = options["release_area_layer_id"], id = options["id"])
    ogr_dataset_release_area = gdal.OpenEx(release_area, gdal.OF_VECTOR)

    # actinia requires input URLs to be quoted if eg & is used
    if not ogr_dataset_release_area:
        ogr_dataset_release_area = gdal.OpenEx(
            parse.unquote(release_area), gdal.OF_VECTOR
        )
    layer_release_area = ogr_dataset_release_area.GetLayerByIndex(0)
    release_extent = layer_release_area.GetExtent()  # Extent is west, east, south, north
    config = dict(layer_release_area.GetNextFeature())  # first feature contains config attributes

    release_name = f"com1DFAV2{config['id']}"

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
    config["nCPU"] = options["nprocs"]

     # Get entrainment area
    if flags["e"]:
        config = get_shape_file_and_config("ENT", config, options)
       

    # Get resistance area
    if flags["r"]:
        config = get_shape_file_and_config("RES", config, options)

    # Write release area to shape
    gdal.VectorTranslate(
        str(avalanche_dir / f"{release_name}.shp"),
        ogr_dataset_release_area,
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

    com1dfa_results_pd = pd.DataFrame(run_com1dfa(config_dict=config)[3])
  
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
