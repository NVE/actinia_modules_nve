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

# %flag
# % key: p
# % description: Compute class boundaries from percentiles (default is linear breaks)
# %end

# %option
# % key: locations
# % type: string
# % required: no
# % multiple: no
# % key_desc: Base name for maps for the managed locations
# % description: Base name for maps for the managed locations
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

# %option G_OPT_R_INPUT
# % key: subclass_map
# % required: no
# % multiple: no
# %end

# %option G_OPT_F_INPUT
# % key: keepass_file
# % required: no
# % multiple: no
# % key_desc: Input KeePass file to get DB credentials from
# % description: Input KeePass file to get DB credentials from
# %end

# %option G_OPT_R_INPUT
# % key: continuous_subdivision_map
# % required: no
# % multiple: no
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

# %rules
# % excludes: subclass_map,continuous_subdivision_map,class_number,round_to_closest
# % required: subclass_map,continuous_subdivision_map
# % collective: keepass_file,keepass_title
# %end

import os
import sys

from subprocess import PIPE

import numpy as np

import grass.script as gs


def round_to_closest(x, y):
    if not y:
        return x
    return tuple(np.round(np.array(x).astype(float) / y, 0).astype(type(y)) * y)


def keepass_to_env(
    keepass_file, keepass_pwd, title, username_var, password_var, first=True
):
    from pykeepass import PyKeePass

    kp = PyKeePass(keepass_file, password=keepass_pwd)
    entry = kp.find_entries(title=title, first=first)
    os.environ[username_var] = entry.username
    os.environ[password_var] = entry.password
    return None


def main():
    locations = options["locations"]
    continuous_subdivision_map = options["continuous_subdivision_map"]
    subclass_map = options["subclass_map"]
    class_number = options["class_number"]
    method = "percentile" if flags["p"] else None
    round_to_closest_m = options["round_to_closest"]
    password_var = "MSSQLSPATIAL_PWD"
    username_var = "MSSQLSPATIAL_UID"

    if options["keepass_file"]:
        if "KEEPASS_PWD" not in os.environ:
            gs.fatal(
                _(
                    "The KeePass password needs to be provided through environment variable 'KEEPASS_PWD'"
                )
            )
        keepass_to_env(
            options["keepass_file"],
            os.environ["KEEPASS_PWD"],
            options["keepass_title"],
            username_var,
            password_var,
            first=True,
        )

    # Import locations
    Module(
        "v.in.ogr",
        input=options["locations"],
        layer=options["layer"],
        where=options["where"],
        output=locations,
    )

    # Check for user-defined breaks
    # tbd
    range_dict = None

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

    if not subclass_map:
        if not range_dict:
            # Compute breaks
            if method == "percentile":
                univar_percentile = list(
                    np.round(
                        np.linspace(0, 100, num=class_number, endpoint=False)[1:], 0
                    ).astype(int)
                )
                univar_flags = "et"

            else:
                univar_percentile = None
                univar_flags = "t"

            stats = Module(
                "r.univar",
                map=continuous_subdivision_map,
                zones=locations,
                flags=univar_flags,
                stdout_=PIPE,
                percentile=univar_percentile,
            )

            univar_stats = np.genfromtxt(
                stats.outputs.stdout.split("\n"), delimiter="|", names=True, dtype=None
            )

            if method == "percentile":
                range_dict = {
                    s["zone"]: round_to_closest(
                        tuple(s[[f"perc_{perc}" for perc in univar_percentile]]),
                        round_to_closest_m,
                    )
                    for s in univar_stats
                    if not np.isnan(s["max"])
                }
            else:
                range_dict = {
                    s["zone"]: round_to_closest(
                        np.linspace(
                            s["min"], s["max"], num=class_number, endpoint=False
                        )[1:],
                        round_to_closest_m,
                    )
                    for s in univar_stats
                    if not np.isnan(s["max"])
                }

            # Update altitude range per parent_id
            # tbd

        # Create reclassified map for breaks per location
        for area_class in range(class_number - 1):
            # area_class = area_class + 1
            rc_rules = "\n".join(
                [
                    f"{x} = {int(y[area_class])} {area_class + 1}"
                    for x, y in range_dict.items()
                ]
            )
            Module(
                "r.reclass",
                input=locations,
                output=f"{locations}_rc{area_class + 1}",
                rules="-",
                stdin_=rc_rules,
                verbose=True,
                overwrite=True,
            )

        # Create category map to cross with location map
        subclass_map = f"{locations}_rc"
        rc_expression = f"{subclass_map}="
        for area_class in range(class_number):
            if area_class == 0:
                rc_expression += f"if({continuous_subdivision_map} <= {locations}_rc{area_class + 1},{area_class + 1},"
            elif area_class == class_number - 1:
                rc_expression += f"if({continuous_subdivision_map} > {locations}_rc{area_class},{area_class + 1},"
            else:
                rc_expression += f"if({continuous_subdivision_map} > {locations}_rc{area_class}&&{continuous_subdivision_map} <= {locations}_rc{area_class + 1},{area_class + 1},"
        rc_expression += "null()"
        rc_expression += ")" * class_number

        Module(
            "r.mapcalc",
            expression=rc_expression,
            verbose=True,
            overwrite=True,
        )

    # Create final map with sub-location
    Module(
        "r.cross",
        input=[locations, subclass_map],
        output=f"{locations}_classes",
        flags="z",
    )

    if range_dict:
        # Update categories in output map
        from grass.pygrass.raster.category import Category

        categories = Category(f"{locations}_classes")
        categories.read()
        cat_rules = []
        for cat in categories:
            parent_id = int(cat[0].split(";")[0])
            sub_id = int(cat[0].split(";")[1].split("category ")[1]) - 1
            if sub_id == 0:
                cat_rules.append(
                    f"{cat[1]}:{parent_id} <= {range_dict[parent_id][sub_id]}",
                )
            elif sub_id == class_number - 1:
                cat_rules.append(
                    f"{cat[1]}:{parent_id} > {range_dict[parent_id][sub_id -1]}",
                )
            else:
                cat_rules.append(
                    f"{cat[1]}:{parent_id} > {range_dict[parent_id][sub_id -1]} & <= {range_dict[parent_id][sub_id]}",
                )
        Module(
            "r.category",
            rules="-",
            stdin="\n".join(cat_rules),
            map=f"{locations}_classes",
            separator=":",
        )

    # Create output vector map
    Module(
        "r.to.vect",
        input=f"{locations}_classes",
        output=f"{locations}_classes",
        type="area",
        flags="vs",
        column="id",
    )

    # Remove temporary data
    # tbd

    # Write output vector map to DB, get IDs and reclass map
    # tbd (either using pyodbc (WKT/WKB geom) or v.out.ogr)


if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    from grass.pygrass.modules.interface import Module

    sys.exit(main())
