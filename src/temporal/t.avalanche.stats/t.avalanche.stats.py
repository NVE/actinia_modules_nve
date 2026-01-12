#!/usr/bin/env python3

"""
MODULE:    t.avalanche.stats
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Extract avalanche parameters from avalanches detected in Sentinel-1 imagery
COPYRIGHT: (C) 2024 by NVE, Stefan Blumentrath

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

"""

# %Module
# % description: Extract avalanche parameters from avalanches detected in Sentinel-1 imagery
# % keyword: temporal
# % keyword: imagery
# % keyword: satellite
# % keyword: Sentinel-1
# % keyword: avalanche
# % keyword: statistics
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_T_WHERE
# %end

# %option G_OPT_R_ELEV
# %end

# %option G_OPT_R_INPUT
# % key: slope
# % description: Topographic slope in degrees (0-90)
# %end

# %option G_OPT_R_INPUT
# % key: aspect
# % description: Topographic aspect in degrees (-180-180) where 0 is North and -90 is West
# %end

# %option G_OPT_M_DIR
# % key: output
# % description: Path to output directory where Shape files with avalanches and avalanche parameters will be stored
# %end

# %option
# % key: valid_area_range
# % type: integer
# % required: no
# % multiple: yes
# % answer: 2000,200000
# % description: Valide range of avalanche size in m2
# %end

import atexit
import sys
import os
from pathlib import Path
from subprocess import PIPE
from zipfile import ZipFile

import grass.script as gs
import grass.temporal as tgis
from grass.pygrass.modules.interface import Module
from grass.pygrass.vector import VectorTopo

TMP_PREFIX = gs.tempname(12)


def cleanup():
    """Remove all temporary data"""
    # Remove Raster map files
    gs.run_command(
        "g.remove",
        type=["raster", "vector"],
        pattern=f"{TMP_PREFIX}*",
        flags="f",
        quiet=True,
    )


def zip_shape(shape_file):
    """Move shape file components into zip archive in the same directory"""
    # Extract shape file path components
    directory = shape_file.parent
    base_name = shape_file.stem
    # Get only shp, dbf, shx and prj components
    shape_file_parts = [
        file_path
        for file_path in directory.glob(f"{base_name}.*")
        if file_path.suffix.lower() in {".shp", ".dbf", ".shx", ".prj"}
    ]
    # Write to zip-file
    with ZipFile(str(directory / f"{base_name}.zip"), "w") as zf:
        for shape_file_part in shape_file_parts:
            zf.write(shape_file_part, os.path.basename(shape_file_part))
            shape_file_part.unlink()


def process_avalanche_map(avalanche_map_row, **kwargs):
    """Extract avalache statistics for a detection raster map"""

    avalanche_map_id = avalanche_map_row["id"]
    avalanche_map = avalanche_map_id.split("@")[0]
    max_val = gs.raster_info(avalanche_map_id)["max"]
    if max_val > 1:
        gs.warning(
            _(
                "Unexpected maximum value '{max}' encountered in map '{map}'. Skipping..."
            ).format(max=max_val, map=avalanche_map_id)
        )
        return
    if max_val < 1:
        gs.warning(_("No avalanches detected in map {}.").format(avalanche_map_id))
        return
    gs.verbose(_("Processing avalanche map {}").format(avalanche_map))
    t_0 = avalanche_map_row["start_time"]
    t_1 = avalanche_map_row["end_time"]
    # semantic_label = avalanche_map_row["semantic_label"]
    name_components = avalanche_map_row["id"].split("_")
    polarization, sat_geom, direction = (
        name_components[4],
        name_components[6],
        name_components[7],
    )
    valid_area_range = kwargs.get("valid_area_range")
    reclass_map = f"{TMP_PREFIX}_{avalanche_map}_rc"
    gs.use_temp_region()
    Module("g.region", align=avalanche_map, raster=avalanche_map)

    # ToDo: use r.clump to get same ID across raster and vector
    #       allows to
    #         - remove v.to.rast and v.edit step,
    #         - filter areas by size before vectorization, and
    #         - to clump diagonal
    Module(
        "r.mapcalc",
        expression=f"{reclass_map}=int(if({avalanche_map_id}==1,1,null()))",
        overwrite=True,
        quiet=True,
    )
    Module("r.to.vect", input=reclass_map, output=avalanche_map, type="area")
    Module(
        "v.db.addcolumn",
        quiet=True,
        map=avalanche_map,
        columns=(
            "t_0 TEXT, t_1 TEXT,"
            "pol TEXT,"
            "sat_geom INTEGER,"
            "direction TEXT, algoritme TEXT,"
            "dtm_min REAL,dtm_mean REAL,"
            "dtm_max REAL,slp_min REAL,"
            "slp_mean REAL, slp_max REAL,"
            "asp_min REAL, asp_mean REAL,"
            "asp_max REAL"
        ),
    )
    # https://grass.osgeo.org/grass84/manuals/libpython/pygrass.vector.html
    with VectorTopo(avalanche_map, mode="r") as avalanche_vmap:
        # db_connection = avalanche_vmap.dblinks.by_index(0).connection()
        # table_name = avalanche_vmap.dblinks.by_index(0).table_name
        areas_n = avalanche_vmap.number_of("areas")
        for idx, area in enumerate(avalanche_vmap.viter("areas")):
            # Skip islands and areas outside the valid area range
            gs.percent(idx + 1, areas_n, 3)

            # Skip areas without centroid / cat
            if not area.centroid() or not area.centroid().cat:
                continue
            if not valid_area_range[0] < area.area() < valid_area_range[1]:
                gs.verbose(
                    _("Skipping area <{cat}> with {area_m2} m2.").format(
                        cat=area.cat, area_m2=area.area()
                    )
                )
                continue
            tmp_map = f"{TMP_PREFIX}_{area.centroid().cat}_raster"
            area_bbox = area.bbox()
            area_attributes = area.attrs
            area_attributes["algoritme"] = "unet"
            area_attributes["t_0"] = t_0.strftime("%Y%m%dT%H%M%S")
            area_attributes["t_1"] = t_1.strftime("%Y%m%dT%H%M%S")
            area_attributes["direction"] = direction
            area_attributes["sat_geom"] = sat_geom
            area_attributes["pol"] = polarization
            gs.use_temp_region()
            Module(
                "g.region",
                align=avalanche_map,
                n=area_bbox.north,
                s=area_bbox.south,
                e=area_bbox.east,
                w=area_bbox.west,
            )
            Module(
                "v.to.rast",
                input=avalanche_map,
                type="area",
                cats=area.cat,
                use="val",
                value=area.cat,
                output=tmp_map,
                quiet=True,
                overwrite=True,
            )
            for map_type in ["elevation", "slope", "aspect"]:
                prefix = {"elevation": "dem", "slope": "slp", "aspect": "asp"}[map_type]
                rmap = kwargs.get(map_type)
                univar_stats = (
                    Module(
                        "r.univar",
                        map=rmap,
                        zones=tmp_map,
                        flags="t",
                        output="-",
                        stdout_=PIPE,
                        quiet=True,
                    )
                    .outputs.stdout.strip("\n")
                    .split("\n")
                )
                univar_stats = {
                    stat[0]: stat[1]
                    for stat in zip(
                        univar_stats[0].split("|"),
                        (univar_stats[1].split("|")),
                        strict=False,
                    )
                    if "nan" not in stat[1]
                }
                for method in ("min", "mean", "max"):
                    area_attributes[f"{prefix}_{method}"] = univar_stats.get(method)

            # Write new attributes to DB
            area_attributes.commit()
            gs.del_temp_region()

    Module(
        "v.edit",
        quiet=True,
        map=avalanche_map,
        tool="delete",
        where="algoritme IS NULL",
    )
    with VectorTopo(avalanche_map, mode="r") as avalanche_vmap:
        if avalanche_vmap.number_of("areas") <= 0:
            gs.warning(
                _("No valid avalanche areas detected for {}").format(avalanche_map)
            )
            return

    Module(
        "v.out.ogr",
        quiet=True,
        input=avalanche_map,
        output=str(kwargs["output"] / f"{avalanche_map}.shp"),
        format="ESRI_Shapefile",
    )
    zip_shape(kwargs["output"] / f"{avalanche_map}.shp")


def main():
    """Get options and run statistics extraction"""
    # Initialize TGIS
    tgis.init()

    # Output could be defined as directory (G_OPT_M_DIR)
    options["output"] = Path(options["output"])
    options["output"].mkdir(parents=True, exist_ok=True)
    options["valid_area_range"] = list(
        map(float, options.get("valid_area_range").split(",", maxsplit=1))
    )
    avalanche_maps = tgis.open_old_stds(options["input"], type="strds")
    registered_avalanche_maps = avalanche_maps.get_registered_maps(
        where=options["where"]
    )

    if not registered_avalanche_maps:
        gs.warning(
            _("Not avalanche maps selected from STRDS <{}>.").format(options["input"])
        )
        sys.exit(0)
    for avalanche_map in registered_avalanche_maps:
        process_avalanche_map(avalanche_map, **options)


if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    main()
