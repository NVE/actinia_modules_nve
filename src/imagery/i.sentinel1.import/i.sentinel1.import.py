#!/usr/bin/env python3

"""
 MODULE:       i.sentinel1.import
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Pre-process and import Sentinel-1 imagery
 COPYRIGHT:    (C) 2023 by Stefan Blumentrath

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

r.timeseries.locations --v --o  locations="nedboersfelt_flomvarsling" \
    locations_url= \
    layer="dbo.coalesced_region_view" \
    where="parent_id IS NULL AND domain_name = 'FLOMVARSLING'" \
    continuous_subdivision_map=DTM_250m@DTM \
    

"""

# %module
# % description: Pre-process and import Sentinel-1 imagery
# % keyword: import
# % keyword: raster
# % keyword: sattelite
# % keyword: sentinel
# %end

# %option G_OPT_R_OUTPUT
# %end

# %option G_OPT_M_DIR
# % key: directory
# % type: string
# % required: no
# % multiple: no
# % answer:
# % description: Directory containting Sentinel-1 imagery in SAFE format
# %end

# %option G_OPT_M_DIR
# % key: output_directory
# % type: string
# % required: no
# % multiple: no
# % answer:
# % description: Directory for storing geocoded GeoTiffs (imported files are removed if not linked)
# %end

# %option
# % key: dem
# % type: string
# % required: yes
# % multiple: no
# % description: Digital elevation model to use for geocoding (either a path to a GeoTiff or a linked raster map)
# %end

# %option
# % key: filter
# % type: string
# % required: no
# % multiple: no
# % description: Regular expression to filter imagery to process
# %end

# %option
# % key: specle_filter
# % type: string
# % required: no
# % multiple: no
# % options: lee,revised_lee
# % description: Apply specle filter algorithms from ESA SNAP
# %end

# %flag
# % key: o
# % description: Override projection check
# %end

# %flag
# % key: l
# % description: Override projection check
# %end

# %flag
# % key: m
# % description: Link resulting data and read statistics from metadata
# %end

# %flag
# % key: r
# % description: Link resulting data and do not read statistics
# %end

# %rules
# % collective: keepass_file,keepass_title
# %end

import os
import sys

from functools import partial
from subprocess import PIPE

import grass.script as gs


def create_graph_xml(metadata):
    """Create XML file with gpt processing graph"""
    graph_xml = Path()

    graph_xml.write_text()
    return graph_xml


def process_image_file(safe_file):
    """Preprocess and import SAFE file
    return string to register map in TGIS"""
    semantic_label = None

    # Extract data (orbit file, data, timestamp)

    start_time = None

    # Create input XML
    
    # Apply processing graph
    gs.call(["gpt", ])

    # Import resulting geotiff
    Module(
        "r.external" if any([flags["l"], flags["m"], flags["r"]]) else "r.in.gdal",
        flags=flags,
        input=str(geocoded_tif),
        output=geocoded_tif.stem,
        memory=2048,
    )
    Module(
        "r.support",
        flags=flags,
        input=geocoded_tif.stem,
        semantic_label=semantic_label,
    )
    Module(
        "r.timestamp",
        flags=flags,
        input=geocoded_tif.stem,
    )
    return "|".join([geocoded_tif.stem, start_time, semantic_label])


def check_directory(directory, ):
    """Check if directory exists and has required access rights"""
    if directory.exists() and os.access(str(directory), W_OK):
        return 0
    if not directory.exists():
        try:
            directory.mkdir(exists_ok=True, Parents=True)
        except OSError:
            gs.fatal(_("Directory <{dir}>").format(str(directory)))
def main():
    """Do the main work"""
    if not gs.find_program("gpt"):
        gs.fatal(_("gpt commandline tool from ESA SNAP not found on current PATH"))

    input_directory = Path(options["directory"])
    check_directory(input_directory, )
    check_directory(Path(options["output_directory"]), )

    # Check / get input DEM

    # Filter imagery
    input_files = input_directory.glob()




if __name__ == "__main__":
    options, flags = gs.parser()
    # lazy imports
    from grass.pygrass.modules.interface import Module
    from grass.pygrass.vector import VectorTopo

    sys.exit(main())
