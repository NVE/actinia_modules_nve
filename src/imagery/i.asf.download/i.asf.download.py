#!/usr/bin/env python3

"""
MODULE:       i.asf.download
AUTHOR(S):    Yngve Antonsen, Stefan Blumentrath (parallelization and additional checks)
PURPOSE:      Searches and Downloads SAR data from the Alaska Satellite Facility
COPYRIGHT:	(C) 2023 by NVE, Yngve Antonsen

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

# %option G_OPT_F_INPUT
# % key: aoi
# % required: no
# % description: Path to GeoJSON file with the Area Of Interest (aoi) (defaults to computational region)
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

# %option
# % key: check_scenes
# % required: no
# % type: string
# % description: Perform checksum / modification time test
# % label: Perform checksum / modification time test for the given set of scenes
# % options: existing,downloaded,all
# %end

# %option G_OPT_M_NPROCS
# %end

# %option
# % key: scenes
# % type: string
# % required: no
# % multiple: no
# % description: Comma separated list of scenes or file with scenes (one per row)
# % label: Selected scenes to download from ASF
# %end

# %flag
# % key: l
# % description: Only list scenes available
# %end

# %flag
# % key: s
# % description: Skip downloading existing scenes
# %end

# %flag
# % key: w
# % description: Write log file with download results
# %end

# %rules
# % required: output_directory,-l
# % requires: -s,output_directory
# % requires: check_scenes,output_directory
# % excludes: -l,-w
# %end

import hashlib
import json
import os
import sys

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import grass.script as gs


def get_aoi_wkt(geojson_file=None):
    """Extract the Area of Interest AOI from a GeoJSON file and
    return it as a WellKnownText (WKT) polygon
    The input GeoJSON should contain only one geometry"""
    if not geojson_file:
        reg = gs.parse_command("g.region", flags="gl", quiet=True)
        if not reg:
            reg = gs.parse_command("g.region", flags="g", quiet=True)
        coordinate_pairs = (
            f"{reg['w']} {reg['n']}",
            f"{reg['w']} {reg['s']}",
            f"{reg['e']} {reg['s']}",
            f"{reg['e']} {reg['n']}",
            f"{reg['w']} {reg['n']}",
        )
        return f"POLYGON (({','.join(coordinate_pairs)}))"
    if not Path(geojson_file).exists():
        gs.fatal(_("AOI file <{}> not found").format(geojson_file))

    try:
        ogr_dataset = ogr.Open(geojson_file)
    except OSError:
        gs.fatal(_("Failed to open AOI file {}").format(geojson_file))
    if not ogr_dataset:
        gs.fatal(_("Could not read AOI file {}").format(geojson_file))
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


def checksum_test(expected_checksum, _dfile):
    """Test if md5 checksum matches for a file"""
    if not isinstance(expected_checksum, str):
        gs.warning(
            _("Reference checksum for {} not readable. Skipping checksum test").format(
                _dfile.name
            )
        )
        return True
    checksum = hashlib.md5(_dfile.read_bytes()).hexdigest()
    if checksum != expected_checksum:
        gs.verbose(_("Checksum test failed for {}").format(_dfile.name))
        _dfile.unlink()
        gs.verbose(_("{} is deleted").format(_dfile.name))
        return False
    gs.verbose(_("Checksum test OK for {}").format(_dfile.name))
    return True


def check_scene(
    asf_search_result,
    download_path="./",
    check_properties=False,
    skip_existing=False,
    alternative_suffixes=("SAFE"),
):
    """Checks if a scene has been downloaded already
    if requested an md5 checksum test is preformed for zip files
    unpacked content (alternative_suffixes) is checked if it is not older than the zip file

    :param asf_search_result: Single ASFProduct object resulting from a search
    :type asf_search_result: ASFProduct
    :param download_path: path to the directory where scenes are downloaded to
    :type download_path: str
    :param check_properties: boolean indicating if md5 sum of the product should be checked
                             If md5 sum does not match the product is treated as non-existent
    :type check_properties: bool
    """
    download_path = Path(download_path)
    scene_name = asf_search_result.properties["fileName"]
    if (download_path / scene_name).exists():
        if check_properties and skip_existing:
            gs.verbose(
                _("Scene {} found on disk. Performing checksum test").format(scene_name)
            )
            if not checksum_test(
                asf_search_result.properties["md5sum"],
                download_path / scene_name,
            ):
                gs.verbose(_("Checksum test for scene {} failed").format(scene_name))
                return asf_search_result
        elif not skip_existing:
            gs.verbose(
                _("Scene {} found on disk but s-flag not set.").format(scene_name)
            )
            if gs.overwrite():
                gs.verbose(
                    _("Removing scene {} to trigger re-download...").format(scene_name)
                )
                (download_path / scene_name).unlink()
                return asf_search_result
        return None
    for alternative_suffix in alternative_suffixes:
        alternative_format = (
            download_path / f"{Path(scene_name).stem}.{alternative_suffix}"
        )
        if alternative_format.exists():
            if check_properties and skip_existing:
                if (
                    datetime.fromisoformat(
                        asf_search_result.properties["processingDate"].replace(
                            "Z", "+00:00"
                        )
                    ).timestamp()
                    > alternative_format.stat().st_mtime
                ):
                    gs.verbose(
                        _(
                            "Scene {scene} found in {file_format} format but scene at ASF is newer"
                        ).format(scene=scene_name, file_format=alternative_suffix)
                    )
                    if gs.overwrite():
                        gs.verbose(
                            _("Removing scene {} to trigger re-download...").format(
                                scene_name
                            )
                        )
                        gs.try_remove(str(alternative_format))
                    return asf_search_result
            elif not skip_existing:
                gs.verbose(
                    _("Scene {} found on disk but s-flag not set.").format(scene_name)
                )
                if gs.overwrite():
                    gs.verbose(
                        _("Removing scene {} to trigger re-download...").format(
                            scene_name
                        )
                    )
                    gs.try_remove(str(alternative_format))
                return asf_search_result
            return None
    return asf_search_result


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

    token_file = Path(token_file or os.path.expanduser("~/.asf_token"))
    if token_file.exists():
        try:
            asf_token = token_file.read_text(encoding="UTF8").rstrip()
        except OSError as error:
            gs.fatal(
                _("Unable to get token from token-file <{}>").format(str(token_file))
            )
    if not asf_token:
        gs.fatal(
            _(
                "No token for authentication provided. Downloading data is thus not possible.\n"
                "Please provide an authentication token"
            )
        )
    return asf_token


def download_with_checksumtest(asf_search_result, download_path="./", session=None):
    """Download ASFProduct with checksum test"""
    download_path = Path(download_path)
    _dfile = download_path / asf_search_result.properties["fileName"]
    gs.verbose(_("Downloading scene {}").format(_dfile.name))
    asf_search_result.download(path=download_path, session=session)
    i = 0
    checksum_match = False
    while i < 3 and not checksum_match:
        checksum_match = checksum_test(asf_search_result.properties["md5sum"], _dfile)
        if not checksum_match:
            gs.verbose(
                _("Checksum test for {} failed. Trying to download again").format(
                    _dfile.name
                )
            )
            asf_search_result.download(path=download_path, session=session)
            i += 1
    if not checksum_match:
        gs.warning(_("Failed to fully download {}").format(_dfile.name))
        return {"failed_downloads": str(_dfile.name)}
    gs.verbose(_("Successfully downloaded {}").format(_dfile.name))
    return {"downloaded": str(_dfile.name)}


def checkout_results(
    result_set,
    initial_scenes=None,
    print_results=False,
    log_results=False,
    with_scene_check=False,
    download_path=Path("./"),
):
    """Helper function for checking, printing and logging search results"""

    if len(result_set) == 0 and not log_results:
        if initial_scenes:
            gs.info(_("All {} scenes previously downloaded").format(initial_scenes))
        else:
            gs.info(_("No results found with given search criteria"))
        sys.exit()

    if initial_scenes:
        gs.verbose(
            _("Selected {0} if {1} scenes for download").format(
                len(result_set), initial_scenes
            )
        )
    else:
        gs.verbose(
            _(
                "Found {} relevant scenes for download. Checking for existing scenes"
            ).format(len(result_set))
        )

    if print_results:
        print("\n".join([str(result.properties["fileName"]) for result in result_set]))
        sys.exit()
    if log_results:
        download_result = {"failed_downloads": [], "downloaded": []}
        if with_scene_check and len(result_set) > 0:
            for result in result_set:
                for result_type, result_item in result.items():
                    download_result[result_type].append(result_item)
        elif len(result_set) > 0:
            download_result = {
                "downloaded": [result.properties["fileName"] for result in result_set]
            }
        with (
            download_path
            / f"i_asf_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ).open("w", encoding="UTF-8") as log_file:
            json.dump(
                download_result,
                log_file,
                sort_keys=True,
                indent=2,
            )


def main():
    """Search and download data products from ASF"""
    # Extract AOI for geo_search
    aoi = get_aoi_wkt(options["aoi"])

    check_scenes = options["check_scenes"]
    skip = flags["s"] or check_scenes in ["all", "existing"]

    # Set keyword arguments for search
    opts = {
        "platform": options["platform"],
        "beamMode": options["beam_mode"],
        "processingLevel": options["processinglevel"],
        "start": options["start"],
        "end": options["end"],
        "intersectsWith": aoi,
    }

    # Search ASF
    if options["scenes"]:
        # Check if scenes is file input
        scenes_input = Path(options["scenes"])
        if scenes_input.exists() and scenes_input.is_file():
            scenes = scenes_input.read_text(encoding="UTF8").strip().split("\n")
        else:
            scenes = options["scenes"].split(",")
        results = asf.granule_search(scenes)
        # Remove metadata from download
        for file in results:
            if (
                file.properties["fileName"].endswith("iso.xml")
                and file.properties["processingLevel"] == "METADATA_GRD_HD"
            ):
                results.remove(file)
    else:
        results = asf.geo_search(**opts)
    checkout_results(results)
    initial_scene_number = len(results)

    # Check for parallelization
    nprocs = min(int(options["nprocs"]), len(results))

    # Authenticate for download if needed
    if not flags["l"]:
        token = get_asf_token(options["token"])
        try:
            token_session = asf.ASFSession().auth_with_token(token)
        except ValueError:
            gs.fatal(_("Token authentication failed"))

        # Check download directory and create if it does not exist
        download_dir = Path(options["output_directory"])
        if not download_dir.exists():
            download_dir.mkdir(exist_ok=True, parents=True)
            if check_scenes == "existing":
                gs.info(
                    _("Download directory does not exist, no existing scenes to check")
                )
                check_scenes = None
            elif check_scenes == "all":
                check_scenes = "downloaded"

    # Setup file check function
    check_local_scene = partial(
        check_scene,
        download_path=options["output_directory"],
        check_properties=check_scenes in ["existing", "all"],
        skip_existing=skip,
        alternative_suffixes=["SAFE"],
    )

    if nprocs == 1:
        results = [result for result in results if check_local_scene(result)]
        checkout_results(
            results, initial_scenes=initial_scene_number, print_results=flags["l"]
        )
        if check_scenes in ["downloaded", "all"] and not flags["l"]:
            results = [
                download_with_checksumtest(
                    result,
                    download_path=options["output_directory"],
                    session=token_session,
                )
                for result in results
            ]
        elif not flags["l"]:
            asf.ASFSearchResults(results).download(
                path=options["output_directory"],
                processes=nprocs,
                session=token_session,
            )
    else:
        # Match local and remote files
        with Pool(nprocs) as pool:
            results = pool.map(check_local_scene, results)
            results = [result for result in results if result]

        checkout_results(
            results, initial_scenes=initial_scene_number, print_results=flags["l"]
        )
        nprocs = min(int(options["nprocs"]), len(results))
        if check_scenes in ["downloaded", "all"] and not flags["l"]:
            with Pool(nprocs) as pool:
                download_with_checksumtest_parallel = partial(
                    download_with_checksumtest,
                    download_path=options["output_directory"],
                    session=token_session,
                )
                results = pool.map(download_with_checksumtest_parallel, results)
        elif not flags["l"]:
            asf.ASFSearchResults(results).download(
                path=options["output_directory"],
                session=token_session,
                processes=nprocs,
            )

    checkout_results(
        results,
        initial_scenes=initial_scene_number,
        log_results=flags["w"],
        with_scene_check=check_scenes in ["downloaded", "all"],
        download_path=download_dir,
    )


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
                "Can not import GDAL python bindings."
                "Please install it with 'pip install GDAL==$GDAL_VERSION'"
            )
        )

    main()
