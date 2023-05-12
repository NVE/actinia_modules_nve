#!/usr/bin/env python3

"""
 MODULE:       r.timeseries.locations
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Manage locations for time series in NVE time series DB
 COPYRIGHT:    (C) 2023 by Stefan Blumentrath

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
# % description: Manage locations for time series in NVE time series DB
# % keyword: NVE
# % keyword: import
# % keyword: export
# % keyword: raster
# % keyword: vector
# % keyword: category
# % keyword: reclass
# % keyword: database
# %end

# %option G_OPT_R_OUTPUT
# % key: locations
# % key_desc: Map name for the managed locations
# % description: Map name for the managed locations
# %end

# %option G_OPT_R_OUTPUT
# % key: locations_subunits
# % required: no
# % key_desc: Map name for subunits of the managed locations
# % description: Map name for subunits of the managed locations
# %end

# %option
# % key: locations_url
# % type: string
# % required: yes
# % multiple: no
# % answer:
# % description: URL to the OGR data source to import
# %end

# %option
# % key: layer
# % type: string
# % required: no
# % multiple: no
# % description: Layer name of the OGR data source to import
# %end

# %option G_OPT_DB_WHERE
# %end

# %option
# % key: snap
# % type: double
# % required: no
# % multiple: no
# % answer: -1
# % description: Snap vertices in input vector map to this number of map units
# %end

# %option
# % key: method
# % options: percentile,linear,database
# % required: no
# % multiple: no
# % description: Method for generating subunits
# %end

# %option G_OPT_F_INPUT
# % key: keepass_file
# % required: no
# % multiple: no
# % key_desc: Input KeePass file to get DB credentials from
# % description: Input KeePass file to get DB credentials from
# %end

# %option
# % key: keepass_title
# % type: string
# % required: no
# % multiple: no
# % description: Title of the KeePass entry to get MS SQL credentials from
# %end

# %option G_OPT_R_INPUT
# % key: continuous_subdivision_map
# % required: no
# % multiple: no
# %end

# %option
# % key: domain_id
# % type: integer
# % required: yes
# % multiple: no
# % description: Domain ID of locations to import
# %end

# %option
# % key: class_number
# % type: integer
# % required: no
# % multiple: no
# % description: Number of classes for subdividing locations
# %end

# %option
# % key: round_to_closest
# % type: string
# % required: no
# % multiple: no
# % description: Closest unit to round class boundaries to (integer or float)
# %end

# %option G_OPT_M_NPROCS
# %end

# %rules
# % collective: keepass_file,keepass_title
# % collective: locations_subunits,method,continuous_subdivision_map
# %end

import os
import sys

from functools import partial
from subprocess import PIPE

import pyodbc
import numpy as np

import grass.script as gs


def round_to_closest(x, y):
    """Round value x to closest y"""
    if not y:
        return x
    return tuple(np.round(np.array(x).astype(float) / y, 0).astype(type(y)) * y)


def keepass_to_env(
    keepass_file, keepass_pwd, title, username_var, password_var, first=True
):
    """Write KeePass entries into environment variables"""
    from pykeepass import PyKeePass

    kp = PyKeePass(keepass_file, password=keepass_pwd)
    entry = kp.find_entries(title=title, first=first)
    os.environ[username_var] = entry.username
    os.environ[password_var] = entry.password


def create_graph(raster_map, values, min_value="-Inf", max_value="Inf", epsilon=0.00000001):
    """Create a map calculator graph() function from a mapname and a list of range values
    :param raster_map: Name of the raster map to create the graph function for
    :type raster_map: str
    :param values: tuple with break points to build the graph with
    :type values: tuple
    :param min: Minimum value of the input raster map
    :type min: float
    :param max: Maximum value of the input raster map
    :type max: float
    :param epsilon: minimal value to dstigush break boundaries
    :type epsilon: float
    :return: A string with a graph() function for r.mapcalc
    """
    value_list = []
    for idx, value in enumerate(values):
        if idx == 0:
            value_list.append(f"{min_value},{value[0]},{value[2] - epsilon},{value[0]}")
        elif idx == len(values) - 1:
            value_list.append(f"{value[1]},{value[0]},{max_value},{value[0]}")
        else:
            value_list.append(f"{value[1]},{value[0]},{value[2] - epsilon},{value[0]}")

    return f"graph({raster_map},{','.join(value_list)})"


def range_dict_from_db(grass_options):
    """Read user defined breaks for each ID from DB
    :param options: The GRASS GIS options dict from g.parser
    :type options: dict
    :return: A dict with class breaks per ID
    """
    # Get data from DB
    conn = pyodbc.connect(
        grass_options["locations_url"].replace("MSSQL:", "")
        + f";UID={os.environ.get('MSSQLSPATIAL_UID')};PWD={os.environ.get('MSSQLSPATIAL_PWD')}"
    )
    cursor = conn.cursor()
    res = cursor.execute(
        f"""SELECT parent_id, id, minimum_elevation_m, maximum_elevation_m
  FROM {grass_options["layer"]}
  WHERE domain_id = {grass_options["domain_id"]} AND parent_id IS NOT NULL
  ORDER BY parent_id, minimum_elevation_m, maximum_elevation_m
;"""
    )
    range_table = res.fetchall()
    conn.close()
    # Build range dict
    range_dict = {}
    for row in range_table:
        if row[0] in range_dict:
            range_dict[row[0]].append(row[1:])
        else:
            range_dict[row[0]] = [(row[1:])]

    return range_dict


def range_dict_from_statistics(grass_options):
    """Compute breaks from statistics of the continuous_subdivision_map
    according to user given method, class_number and rounding precision
    :param options: The GRASS GIS options dict from g.parser
    :type options: dict
    :return: A dict with class breaks per ID
    """
    class_number = int(grass_options["class_number"])
    # Compute breaks
    if grass_options["method"] == "percentile":
        univar_percentile = list(
            np.round(
                np.linspace(0, 100, num=class_number + 1, endpoint=True), 0
            ).astype(int)
        )
        univar_flags = "et"

    elif grass_options["method"] == "linear":
        univar_percentile = None
        univar_flags = "t"

    stats = Module(
        "r.univar",
        map=grass_options["continuous_subdivision_map"],
        zones=grass_options["locations"],
        flags=univar_flags,
        stdout_=PIPE,
        percentile=univar_percentile,
    )

    univar_stats = np.genfromtxt(
        stats.outputs.stdout.split("\n"),
        delimiter="|",
        names=True,
        dtype=None,
        encoding="UTF8",
    )

    if grass_options["method"] == "percentile":
        range_dict = {
            s["zone"]: round_to_closest(
                tuple(s[[f"perc_{perc}" for perc in univar_percentile]]),
                float(grass_options["round_to_closest"]),
            )
            for s in univar_stats
            if not np.isnan(s["max"])
        }
    elif grass_options["method"] == "linear":
        range_dict = {
            int(s["zone"]): list(
                round_to_closest(
                    np.linspace(
                        s["min"], s["max"], num=class_number + 1, endpoint=True
                    ),
                    float(grass_options["round_to_closest"]),
                )
            )
            for s in univar_stats
            if not np.isnan(s["max"])
        }
    return {
        key: [
            (key * 10 + idx + 1, v, val[idx + 1])
            for idx, v in enumerate(val)
            if idx + 2 < len(val)
        ]
        for key, val in range_dict.items()
    }


def main():
    """Do the main work"""
    locations = options["locations"]
    where = f"domain_id = {options['domain_id']} AND parent_id IS NULL"
    continuous_subdivision_map = options["continuous_subdivision_map"]
    schema, layer = options["layer"].split(".")

    if options["keepass_file"]:
        if "KEEPASS_PWD" not in os.environ:
            gs.fatal(
                _(
                    "The KeePass password needs to be provided through environment variable 'KEEPASS_PWD'"
                )
            )
        gs.verbose(
            _("Trying to get keepass entries from <{}>").format(options["keepass_file"])
        )
        keepass_to_env(
            options["keepass_file"],
            os.environ["KEEPASS_PWD"],
            options["keepass_title"],
            "MSSQLSPATIAL_UID",
            "MSSQLSPATIAL_PWD",
            first=True,
        )

    # Import locations
    Module(
        "v.in.ogr",
        flags="o",
        # Until GDAL 3.6 is available UID and PWD have to be provided in the connection string
        input=options["locations_url"]
        + f";Tables={schema}.{layer};UID={os.environ.get('MSSQLSPATIAL_UID')};PWD={os.environ.get('MSSQLSPATIAL_PWD')}",
        layer=layer if schema == "dbo" else f"{schema}.{layer}",
        where=where + " AND " + options["where"] if options["where"] else where,
        output=locations,
        snap=options["snap"],
    )
    gs.vector.vector_history(locations, replace=True)

    # Set computational region
    Module(
        "g.region",
        vector=locations,
        align=continuous_subdivision_map,
        flags="g",
    )

    # Convert loactions to raster
    Module(
        "v.to.rast",
        input=locations,
        output=locations,
        type="area",
        use="attr",
        attribute_column="id",
        label_column="name",
        memory=2048,
    )
    gs.raster.raster_history(locations, overwrite=True)

    if options["locations_subunits"]:
        if options["method"] == "database":
            range_dict = range_dict_from_db(options)
        elif options["method"] in ["percentile", "linear"]:
            range_dict = range_dict_from_statistics(options)

        create_sub_graph = partial(
            create_graph,
            continuous_subdivision_map,
            min_value=-9999,
            max_value=9999,
            epsilon=0.000001,
        )
        mc_expression = f"""{options["locations_subunits"]}=int(graph({locations},{", ".join(f"{cat}, int({create_sub_graph(values)})" for cat, values in range_dict.items())}))"""

        if int(options["nprocs"]) > 1:
            Module(
                "r.mapcalc.tiled",
                expression=mc_expression,
                nprocs=int(options["nprocs"]),
            )
        else:
            Module("r.mapcalc", expression=mc_expression)

        # Add category labels
        Module(
            "r.category",
            map=options["locations_subunits"],
            rules="-",
            separator="tab",
            stdin_="\n".join(
                [
                    "\n".join([f"{row[0]}\t{row[1]} - {row[2]}" for row in val])
                    for val in range_dict.values()
                ]
            ),
        )
        gs.raster.raster_history(options["locations_subunits"], overwrite=True)

        # Create output vector map
        Module(
            "r.to.vect",
            input=options["locations_subunits"],
            output=options["locations_subunits"],
            type="area",
            flags="vs",
            column="id",
        )
        gs.vector.vector_history(options["locations_subunits"], replace=True)

        # Get geometries as WKT
        vector_map = VectorTopo(options["locations_subunits"])
        vector_map.open("r")

        geom_dict = {}
        for parent_id in range_dict.values():
            for area_id in parent_id:
                geom_dict[area_id[0]] = ""
        geom_dict[None] = ""

        for area in vector_map.viter("areas"):
            geom_dict[area.cat] += area.to_wkt().replace("POLYGON", "")

        vector_map.close()

        # Load geometries of subunits to MS SQL
        conn = pyodbc.connect(
            options["locations_url"].replace("MSSQL:", "")
            + f";UID={os.environ.get('MSSQLSPATIAL_UID')};PWD={os.environ.get('MSSQLSPATIAL_PWD')}"
        )
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE [dbo].[region] SET geom = geometry::STGeomFromText(?, 25833) WHERE id = ?",
            [
                (f"MULTIPOLYGON ({polygons.replace(') (', '), (')})", polygon_id)
                for polygon_id, polygons in geom_dict.items()
                if polygon_id and polygons
            ],
        )
        cursor.commit()
        conn.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    from grass.pygrass.modules.interface import Module
    from grass.pygrass.vector import VectorTopo

    sys.exit(main())
