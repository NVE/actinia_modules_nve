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

# %option G_OPT_F_OUTPUT
# % description: Output GeoPackage file with avalanches and avalanche parameters
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
from pathlib import Path
from subprocess import PIPE

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


def process_avalanche_map(avalanche_map_row, **kwargs):
    """Extract avalache statistics for a detection raster map"""

    avalanche_map_id = avalanche_map_row["id"]
    avalanche_map = avalanche_map_id.split("@")[0]
    gs.verbose(_("Processing avalanche map {}").format(avalanche_map))
    start_time = avalanche_map_row["start_time"]
    end_time = avalanche_map_row["end_time"]
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
    Module(
        "r.reclass",
        input=avalanche_map_id,
        output=reclass_map,
        rules="-",
        stdin_="1 = 1\n* = NULL",
    )
    Module("r.to.vect", input=reclass_map, output=avalanche_map, type="area")
    # Module("v.to.db", input=avalanche_map, method=f"{TMP_PREFIX}_clumped_rc", column="areal_m2")
    Module(
        "v.db.addcolumn",
        map=avalanche_map,
        columns=(
            "start_time TEXT, end_time TEXT,"
            "polarization TEXT,"
            "sat_geom INTEGER, polarization TEXT,"
            "direction TEXT, algoritme TEXT,"
            "dtm_min REAL,dtm_mean REAL,"
            "dtm_max REAL,slope_min REAL,"
            "slope_mean REAL, slope_max REAL,"
            "aspect_min REAL, aspect_mean REAL,"
            "aspect_max REAL"
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
            if (
                area.area() < valid_area_range[0]
                or area.area() > valid_area_range[1]
                or not area.cat
            ):
                gs.verbose(
                    _("Skipping area <{cat}> with {area_m2} m2.").format(
                        cat=area.cat, area_m2=area.area()
                    )
                )
                continue
            tmp_map = f"{TMP_PREFIX}_{area.cat}_raster"
            area_bbox = area.bbox()
            area_attributes = area.attrs
            area_attributes["algoritme"] = "unet"
            area_attributes["start_time"] = start_time.strftime("%Y%m%dT%H%M%S")
            area_attributes["end_time"] = end_time.strftime("%Y%m%dT%H%M%S")
            # area_attributes["polarization"] = polarization
            area_attributes["direction"] = direction
            area_attributes["sat_geom"] = sat_geom
            area_attributes["polarization"] = polarization
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
                prefix = "dtm" if map_type == "elevation" else map_type
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

    Module("v.edit", map=avalanche_map, tool="delete", where="algoritme IS NULL")
    export_module = Module(
        "v.out.ogr",
        input=avalanche_map,
        output=str(kwargs["output"]),
        format="GPKG",
        run_=False,
    )
    if kwargs["output"].exists():
        export_module.flags.u = True
        export_module.overwrite = True
        # export_module.inputs.output_layer = avalanche_map
    export_module.run()


def main():
    """Get options and run statistics extraction"""
    # Initialize TGIS
    tgis.init()

    # Output could be defined as directory (G_OPT_M_DIR)
    options["output"] = Path(
        options["output"]
        if options["output"].endswith(".gpkg")
        else f"{options['output']}.gpkg"
    )
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
