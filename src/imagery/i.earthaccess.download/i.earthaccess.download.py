#!/usr/bin/env python3

"""MODULE:    i.earthaccess.download
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Searches and Downloads earth observation data using the
           EarthAccess library for NASA Earthdata APIs.
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
# % description: Searches and downloads earth observation data using the EarthAccess library for NASA Earthdata APIs.
# % keyword: imagery
# % keyword: satellite
# % keyword: download
# % keyword: EarthData
# % keyword: earthaccess
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
# % required: no
# % type: string
# % description: GRASS GIS Map or OGR readable file with one polygon to delinate the Area of Interest (AOI)
# % guisection: Filter
# %end

# %option
# % key: temporal
# % required: no
# % type: string
# % multiple: yes
# % description: Pair of ISO-formated time stamps (YYYY-MM-DD HH:MM:SS) for start end end of sensing time
# % guisection: Filter
# %end

# %option
# % key: created_at
# % required: no
# % type: string
# % multiple: yes
# % description: Pair of ISO-formated time stamps (YYYY-MM-DD HH:MM:SS) for start end end of creation time
# % guisection: Filter
# %end

# %option
# % key: production_date
# % required: no
# % type: string
# % multiple: yes
# % description: Pair of ISO-formated time stamps (YYYY-MM-DD HH:MM:SS) for start end end of production time
# % guisection: Filter
# %end

# %option
# % key: revision_date
# % required: no
# % type: string
# % multiple: yes
# % description: Pair of ISO-formated time stamps (YYYY-MM-DD HH:MM:SS) for start end end of modification time
# % guisection: Filter
# %end

# %option
# % key: keyword
# % required: no
# % type: string
# % description: Keyword used for searching datasets or collection (supports wildcards)
# %end

# %option
# % key: short_name
# % required: no
# % type: string
# % description: Short name of dataset to download or collection to search
# %end

# %option
# % key: granule_name
# % required: no
# % type: string
# % description: File name of the granule to download (supports wildcards)
# %end

# %option
# % key: provider
# % required: no
# % type: string
# % description: Provider to download from
# %end

# %option
# % key: limit
# % required: no
# % type: integer
# % description: Limit number of matches to return / download
# %end

# %option
# % key: print
# % required: no
# % type: string
# % description: Print search result (do not download)
# % options: collections,collection_names,granule_metadata
# %end

# %option G_OPT_F_FORMAT
# % key: format
# % description: Print search result (do not download)
# %end

# %option G_OPT_F_OUTPUT
# % key: file
# % required: no
# % description: Write search result to file
# %end

# %option G_OPT_M_NPROCS
# % key: nprocs
# % description: Number of cores used for downloading
# %end

# %option
# % key: check_scenes
# % required: no
# % type: string
# % description: Perform checksum / modification time test
# % label: Perform checksum / modification time test for the given set of scenes
# % options: existing,downloaded,all
# %end

# %option
# % key: scenes
# % type: string
# % required: no
# % multiple: no
# % description: Comma separated list of scenes or file with scenes (one per row)
# % label: Selected scenes to download using earthaccess
# %end

# %option
# % key: order_by
# % type: string
# % required: no
# % multiple: yes
# % description: Comma separated list of sort parameters
# % options: temporal,revision_date
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
# % required: keyword,short_name
# %end

import json
import sys
from datetime import datetime
from pathlib import Path

import grass.script as gs
from grass.pygrass.vector import VectorTopo


def get_spatial_query_parameter(aoi: str) -> dict:
    """Generate the spatial query parameter from user input

    The input aoi (=area of interest) can be:
     a) a valid GeoJSON
     b) a GRASS GIS vector map
     c) None
    Supported spatial search options in earthsearch are:
    - point (lon, lat)
    - circle (lon, lat, dist)
    - polygon (coordinate sequence)
    - bounding_box (ll_lon, ll_lat, ur_lon, ur_lat)
    - line (coordinate sequence)

    Currently only polygon and bounding_box are supported in this module.

    :param aoi: Name of GRASS GIS vector map or path to OGR readable dataset
                with one polygon
    :type aoi: str
    :param download_path: path to the directory where scenes are downloaded to

    :returns spatial_filters: dict with filter type and coodinates
    :rtype spatial_filter: dict

    """
    # Check if location is latlon
    reg = gs.parse_command("g.region", flags="gl", quiet=True)

    if not aoi:
        gs.debug(_("Using the bounding box from computational region as AOI"))
        # Use bounding box from computational region
        if reg:
            return {
                "bounding_box": (
                    min(float(reg["sw_long"]), float(reg["nw_long"])),
                    (min(float(reg["sw_lat"]), float(reg["se_lat"]))),
                    max(float(reg["se_long"]), float(reg["ne_long"])),
                    (max(float(reg["nw_lat"]), float(reg["ne_lat"]))),
                ),
            }
        reg = gs.parse_command("g.region", flags="g", quiet=True)
        return {"bounding_box": (reg["w"], reg["s"], reg["e"], reg["n"])}

    transform_coordinates = None
    wgs_84 = osr.SpatialReference()
    wgs_84.ImportFromEPSG(4326)
    wgs_84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Try GeoJSON / AOI file with OGR
    if Path(aoi).exists():
        gs.debug(_("Reading AOI from file {}").format(aoi))
        try:
            ogr_dataset = ogr.Open(aoi)
        except OSError:
            gs.fatal(_("Failed to open AOI file {}").format(aoi))
        if not ogr_dataset:
            gs.fatal(_("Could not read AOI file {}").format(aoi))
        if ogr_dataset.GetLayerCount() > 1:
            gs.warning(_("Input file contains more than one layer"))
        ogr_layer = ogr_dataset.GetLayerByIndex(0)
        if ogr_layer.GetGeomType() != 3:
            gs.warning(_("GeoJSON does not contain polygons"))
        if ogr_layer.GetFeatureCount() > 1:
            gs.warning(
                _("GeoJSON contains more than one geometry. Using only the first one."),
            )
        layer_crs = ogr_layer.GetSpatialRef()
        ogr_feature = ogr_layer.GetFeature(0)
        geom = ogr_feature.GetGeometryRef()
        if not layer_crs.IsSame(wgs_84):
            geom.TransformTo(wgs_84)
        ring = geom.GetGeometryRef(0)

        return {"polygon": ring.GetPoints()}

    # Try GRASS GIS vector map
    if gs.legal_name(aoi):
        aoi_map = VectorTopo(aoi)
        if aoi_map.exist():
            aoi_map.open("r")
            if aoi_map.number_of("areas") > 1:
                gs.warning(
                    _(
                        "GeoJSON contains more than one geometry. Using only the first one.",
                    ),
                )
            if aoi_map.number_of("areas") > 1:
                gs.warning(
                    _(
                        "GeoJSON contains more than one geometry. Using only the first one.",
                    ),
                )
            area = aoi_map.viter("areas")
            coordinate_pairs = area.points().to_list()

            srs_crs = osr.SpatialReference()
            srs_crs.ImportFromWkt(gs.read_command("g.proj", flags="w"))
            if not srs_crs.IsSame(wgs_84):
                transform_coordinates = osr.CoordinateTransformation(srs_crs, wgs_84)
                coordinate_pairs = tuple(
                    tuple(point[0:2])
                    for point in transform_coordinates.TransformPoints(coordinate_pairs)
                )
            else:
                coordinate_pairs = tuple(coordinate_pairs)
            return {"polygon": coordinate_pairs}
        gs.debug(_("AOI vector map <{}> not found.").format(aoi))
    gs.fatal(_("Invalid input for AOI option"))


def get_temporal_query_parameters(user_options: dict) -> dict:
    """Extract temporal query parameters from user given module options."""
    temporal_filters = {}
    # Set keyword arguments with temporal range for search
    for search_option in ("temporal", "production_date", "created_at", "revision_date"):
        if not user_options[search_option]:
            continue
        filter_values = user_options[search_option].split(",")
        if not filter_values:
            continue
        if len(filter_values) > 2:
            gs.fatal(_("Too many input values for <{}>. It cannot be more than two."))
        try:
            filter_values = tuple(
                datetime.fromisoformat(time_stamp) for time_stamp in filter_values
            )
        except:
            gs.fatal(
                _(
                    "Invalid input for <{}>. It must be a sing or pair of ISO-formated datetime(s)",
                ),
            )
        if len(list(filter_values)) < 2:
            filter_values = (filter_values[0], None)
        temporal_filters[search_option] = filter_values

    return temporal_filters


def extract_core_umm_metadata(dataset_dict: dict) -> dict:
    umm_keys = {
        "Abstract",
        "AccessConstraints",
        "AdditionalAttributes",
        "AncillaryKeywords",
        "ArchiveAndDistributionInformation",
        "AssociatedDOIs",
        "CollectionCitations",
        "CollectionDataType",
        "CollectionProgress",
        "ContactGroups",
        "ContactPersons",
        "DOI",
        "DataCenters",
        "DataDates",
        "DataLanguage",
        "DirectDistributionInformation",
        "DirectoryNames",
        "EntryTitle",
        "ISOTopicCategories",
        "LocationKeywords",
        "MetadataAssociations",
        "MetadataDates",
        "MetadataLanguage",
        "MetadataSpecification",
        "PaleoTemporalCoverages",
        "Platforms",
        "ProcessingLevel",
        "Projects",
        "PublicationReferences",
        "Purpose",
        "Quality",
        "RelatedUrls",
        "ScienceKeywords",
        "ShortName",
        "SpatialExtent",
        "SpatialInformation",
        "StandardProduct",
        "TemporalExtents",
        "TemporalKeywords",
        "TilingIdentificationSystems",
        "UseConstraints",
        "Version",
        "VersionDescription",
    }

    def _get_spatial_extent(dataset_dict: dict) -> str:
        """Gsr = set()
        ...: for d in datasets:
        ...:     # keys.update(list(d["umm"].keys()))
        ...:     # gsr.update(d["umm"].get('SpatialExtent').keys())
        ...:     spatial_representation = d["umm"].get('SpatialExtent').get('GranuleSpatialRepresentation')
        ...:     if spatial_representation == 'CARTESIAN':
        ...:         spatial_domain = d["umm"].get('SpatialExtent').get('HorizontalSpatialDomain')
        ...:         if spatial_domain:
        ...:             {'Geometry', 'ResolutionAndCoordinateSystem', 'ZoneIdentifier'}
        ...:             if d["umm"].get('SpatialExtent').get('HorizontalSpatialDomain').get('Geometry'):
        ...:                 gsr.update(d["umm"].get('SpatialExtent').get('HorizontalSpatialDomain').get('Geometry').keys())
        ...:             print(d["umm"].get('SpatialExtent').get('HorizontalSpatialDomain').get('ResolutionAndCoordinateSystem'))
        ...:             print(d["umm"].get('SpatialExtent').get('HorizontalSpatialDomain').get('ZoneIdentifier'))
        ...:
        """
        spatial_dict = {}
        spatial_representation = (
            dataset_dict["umm"].get("SpatialExtent").get("GranuleSpatialRepresentation")
        )
        if spatial_representation == "CARTESIAN":
            print(
                dataset_dict["umm"].get("SpatialExtent").get("HorizontalSpatialDomain"),
            )
        elif spatial_representation in {"GEODETIC", "NO_SPATIAL", "ORBIT"}:
            pass
        return (
            dataset_dict["umm"]
            .get("SpatialExtent")
            .get("HorizontalSpatialDomain")
            .get("Geometry")
        )

    def _get_temporal_extent(dataset_dict: dict) -> tuple:
        """Return the temporal extent of a granule / dataset."""
        datetime_range = (
            dataset_dict["umm"].get("TemporalExtents")[0].get("RangeDateTimes", "")
        )
        if datetime_range:
            return datetime_range[0].get("BeginningDateTime", ""), datetime_range[
                0
            ].get("BeginningDateTime", "")
        return "", ""

    def _get_temporal_extent(dataset_dict: dict) -> tuple:
        """Return the temporal extent of a granule / dataset."""
        datetime_range = (
            dataset_dict["umm"].get("TemporalExtents")[0].get("RangeDateTimes", "")
        )
        if datetime_range:
            return datetime_range[0].get("BeginningDateTime", ""), datetime_range[
                0
            ].get("BeginningDateTime", "")
        return "", ""

    def _get_doi(dataset_dict: dict) -> str:
        """Return DOI of the dataset if available, otherwise return empty string."""
        return (
            dataset_dict["umm"].get("DOI").get("Authority", "")
            + "/"
            + dataset_dict["umm"].get("DOI").get("DOI", "")
        )

    def _get_iso_categories(dataset_dict: dict) -> str:
        iso_cats = dataset_dict["umm"].get("ISOTopicCategories")
        return "|".join(iso_cats) if iso_cats else ""

    return {
        "ShortName": dataset_dict["umm"].get("ShortName", ""),
        "Version": dataset_dict["umm"].get("Version", ""),
        "EntryTitle": dataset_dict["umm"].get("EntryTitle", ""),
        "Abstract": dataset_dict["umm"].get("Abstract", ""),
        "Purpose": dataset_dict["umm"].get("Purpose", ""),
        "ProcessingLevel": dataset_dict["umm"].get("ProcessingLevel").get("Id", ""),
        "TemporalExtent": _get_temporal_extent(dataset_dict),
        "TemporalExtent_1": _get_doi(dataset_dict),
        "TemporalExtent_2": _get_iso_categories(dataset_dict),
        "TemporalExtent_3": _get_spatial_extent(dataset_dict),
    }


def main():
    """Search and download data products using earthaccess API."""
    check_scenes = options["check_scenes"]
    skip = flags["s"] or check_scenes in ["all", "existing"]

    # Extract AOI for geo_search
    search_options = get_spatial_query_parameter(options["aoi"])

    # Extract temporal search criteria
    search_options.update(get_temporal_query_parameters(options))

    # Extract other search criteria
    search_options.update(
        {
            search_option: options[search_option]
            for search_option in ("provider", "keyword", "short_name", "granule_name")
            if options[search_option]
        },
    )

    if options["limit"]:
        search_options["count"] = int(options["limit"])

    # Check download directory and create if it does not exist
    download_dir = Path(options["output_directory"])
    if not download_dir.exists():
        download_dir.mkdir(exist_ok=True, parents=True)
        if check_scenes == "existing":
            gs.info(_("Download directory does not exist, no existing scenes to check"))
            check_scenes = None
        elif check_scenes == "all":
            check_scenes = "downloaded"

    # Try login to earthaccess
    try:
        earthaccess.login()
    except Exception:
        gs.warning(
            _(
                "Login to EarthData failed. Download may fail or search may return incomplete results.",
            ),
        )

    # https://github.com/nsidc/earthaccess/blob/0385d126695807f5c865076350b7def04109e088/earthaccess/api.py#L35
    # print options: collections,dataset_names,granule_metadata
    if options["print"] == "collections" or options["print"] == "collection_names":
        if "keyword" not in search_options and "short_name" not in search_options:
            search_options["keyword"] = "*"
        try:
            datasets = earthaccess.search_datasets(**search_options)
        except:
            gs.fatal(
                _(
                    "Collection search failed. Please check the search parameters and login information.",
                ),
            )
        if options["print"] == "collections":
            if options["format"] == "json":
                print(datasets)
            else:
                print(
                    "\n".join(
                        [
                            "|".join(
                                [
                                    d["umm"].get("ShortName", ""),
                                    d["umm"].get("EntryTitle", ""),
                                    d["umm"].get("Abstract", ""),
                                    d["umm"].get("Purpose", ""),
                                    d["umm"].get("ProcessingLevel").get("Id", ""),
                                    (
                                        d["umm"].get("DOI").get("Authority")
                                        + "/"
                                        + d["umm"].get("DOI").get("DOI")
                                        if d["umm"].get("DOI").get("Authority")
                                        else ""
                                    ),
                                ],
                            )
                            for d in datasets
                        ],
                    ),
                )
            sys.exit(0)
        else:  # if options["print"] == "collection_names":
            collection_names = [d["umm"].get("ShortName", "") for d in datasets]
            if options["format"] == "json":
                print(json.dumps(collection_names, indent=2))
            else:
                print("\n".join(collection_names))
            sys.exit(0)

    if "keyword" in search_options:
        search_options.pop("keyword")
        gs.warning(
            _("'keyword' is not a supported parameter for granule search. Ignoring..."),
        )

    # https://github.com/podaac/tutorials/blob/master/notebooks/SearchDownload_SWOTviaCMR.ipynb
    data_granules = earthaccess.search_data(**search_options)

    if options["print"] == "granule_metadata":
        if format == "json":
            print(json.dumps(data_granules, indent=2))
        else:
            print(data_granules)
        sys.exit(0)

    if flags["s"]:
        for file in data_granules:
            if (
                file.properties["fileName"].endswith("iso.xml")
                and file.properties["processingLevel"] == "METADATA_GRD_HD"
            ):
                data_granules.remove(file)

    nprocs = int(options["nprocs"])
    gs.verbose(
        _("Start downloading {n} granules using {p} threads.").format(
            n=len(data_granules),
            p=nprocs,
        ),
    )

    try:
        earthaccess.download(data_granules, download_dir, threads=nprocs)
    except:
        gs.fatal(
            _(
                "Downloading data failed. Please check search parameters and login information.",
            ),
        )


if __name__ == "__main__":
    options, flags = gs.parser()

    try:
        import earthaccess
    except ImportError:
        gs.fatal(
            _(
                "Can not import the earthaccess library. "
                "Please install it with 'pip install earthaccess'",
            ),
        )

    try:
        from osgeo import ogr, osr

        ogr.UseExceptions()
        osr.UseExceptions()
    except ImportError:
        gs.fatal(
            _(
                "Can not import the GDAL python library. "
                "Please install it with 'pip install GDAL==$(gdal-config --version)'",
            ),
        )

    main()
