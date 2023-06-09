#!/usr/bin/env python3

############################################################################
#
# MODULE:       i.asf.download
#
# AUTHOR(S):    Yngve Antonsen
#
# PURPOSE:      Searches and Downloads SAR data from the Alaska Satellite Facility
#
#
# COPYRIGHT:	(C) 2023 by NVE, Yngve Antonsen
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#############################################################################

# %Module
# % description: Searches and downloads SAR data from the Alaska Satellite Facility.
# % keyword: imagery
# % keyword: satellite
# % keyword: download
# % keyword: SAR
# % keyword: Sentinel
# %end

# %option
# % key: output_directory
# % type: string
# % required: no
# % description: Name for output directory where to store downloaded Sentinel data
# % label: Directory where to store downloaded data
# %end

# %option
# % key: aoi
# % type: string
# % required: yes
# % multiple: no
# % description: Path to GeoJSON file with the Area Of Interest (aoi)
# % label: Path to GeoJSON file with the Area Of Interest (aoi)
# %end

# %option
# % key: token
# % type: string
# % required: no
# % multiple: no
# % description: Path to ASF token file
# % label: File has to contain ASF token
# %end

# %option
# % key: start
# % required: yes
# % type: string
# % description: Start date ('YYYY-MM-DD')
# % guisection: Filter
# %end

# %option
# % key: end
# % required: no
# % type: string
# % description: End date ('YYYY-MM-DD')
# % guisection: Filter
# %end

# %option
# % key: platform
# % required: yes
# % type: string
# % description: Satellite platform
# % label: Currently only Sentinel-1 is supported
# % options: Sentinel-1
# % answer: Sentinel-1
# %end

# %option
# % key: beam_mode
# % required: yes
# % type: string
# % description: Satellite beam mode
# % label: Currently only IW is supported
# % options: IW
# % answer: IW
# %end

# %option
# % key: processinglevel
# % required: no
# % type: string
# % description: SAR processing level
# % label: Ground-Range-Detected (GRD) or Single-Look-Complex (SLC)
# % options: GRD_HD,SLC
# % answer: GRD_HD
# %end

# %flag
# % key: l
# % description: Only list scenes available
# %end

# %flag
# % key: c
# % description: Do checksum test of all scenes
# %end

# %rules
# % required: output_directory,-l
# % excludes: -l,-c
# %end

import os
import hashlib
import sys

from pathlib import Path


import grass.script as gs


def get_aoi_wkt(geojson_file):
    """Extract the Area of Interest AOI from a GeoJSON file and
    return it as a WellKnownText (WKT) polygon
    The input GeoJSON should contain only one geometry"""
    ogr_dataset = ogr.Open(geojson_file)
    if ogr_dataset.GetLayerCount() > 1:
        gs.warning(_("Input file contains more than one layer"))
    ogr_layer = ogr_dataset.GetLayerByIndex(0)
    if ogr_layer.GetGeomType() != 3:
        gs.warning(_("GeoJSON does not contain polygons"))
    if ogr_layer.GetFeatureCount() > 1:
        gs.warning(
            _("GeoJSON contains more than one geometry. Using only the first one.")
        )
    ogr_feature = ogr_layer.GetFeature(0)
    return ogr_feature.geometry().ExportToIsoWkt()


def checksum_test(checksum, expected_checksum, _dfile):
    """ """
    if checksum != expected_checksum:
        gs.verbose(_("Checksum test failed for {}").format(os.path.split(_dfile)[1]))
        os.remove(_dfile)
        gs.verbose(f"{os.path.split(_dfile)[1]} is deleted")
        return False
    gs.verbose(f"Checksum test OK for {os.path.split(_dfile)[1]}")
    return True


def get_asf_token(token_file=None):
    """
    Method to get the ASF token for authentication to ASF.
    Credentials for authentication to ASF can be either given
    using a token_file (default is ~/.asf_token) or by defining
    environment variables:
    :envvar:`ASF_TOKEN`
    The user's :envvar:`HOME` directoy is allways searched for a .asf_token
    token file.

    :param token_file: Path to token file to read
    :type token_file: str
    """
    # Get authentication
    asf_token = os.environ.get("ASF_TOKEN")

    token_file = token_file or os.path.expanduser("~/.asf_token")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="UTF8") as asf_token_file:
                asf_token = asf_token_file.read()
        except OSError as error:
            raise error
    if not asf_token:
        gs.fatal(
            _(
                "No token for authentication provided. Downloading data is thus not possible.\n"
                "Please provide an authentication token"
            )
        )
    return asf_token


def main():
    """ """
    aoi = get_aoi_wkt(options["aoi"])

    opts = {
        "platform": options["platform"],
        "beamMode": options["beam_mode"],
        "processingLevel": options["processinglevel"],
        "start": options["start"],
        "end": options["end"],
        "intersectsWith": aoi,
    }

    results = asf.geo_search(**opts)
    gs.verbose(f"{len(results)} results found")

    if flags["l"]:
        gs.verbose("Listing all scenes found:")
        print("\n".join([str(r.properties["fileName"]) for r in results]))
        gs.verbose("All available scenes are listed.")
        sys.exit()

    token = get_asf_token(options["token"])

    token_session = asf.ASFSession().auth_with_token(token)

    download_dir = Path(options["output_directory"])
    download_dir.mkdir(exist_ok=True, parents=True)
    gs.verbose(_("Downloading to {}").format(str(download_dir)))

    for r in results:
        _dfile = download_dir / r.properties["fileName"]

        r.download(path=download_dir, session=token_session)
        gs.verbose(f"{r.properties['fileName']} is downloaded")

        if flags["c"]:
            i = 0
            while i < 3:
                expected_checksum = r.properties["md5sum"]
                checksum = hashlib.md5(_dfile.read_bytes()).hexdigest()

                if not checksum_test(checksum, expected_checksum, _dfile):
                    gs.verbose(f"Trying to download {os.path.split(_dfile)[1]} again")
                    r.download(path=download_dir, session=token_session)
                    gs.verbose(f"{r.properties['fileName']} is downloaded again")
                    i += 1
                else:
                    i = 3


if __name__ == "__main__":
    options, flags = gs.parser()

    try:
        import asf_search as asf
    except ImportError:
        gs.fatal(
            _(
                "Can not import asf_search. Please install it with 'pip install asf_search'"
            )
        )

    try:
        from osgeo import ogr
    except ImportError:
        gs.fatal(
            _(
                "Can not import GDAL python bindings. Please install it with 'pip install GDAL==$GDAL_VERSION'"
            )
        )

    main()
