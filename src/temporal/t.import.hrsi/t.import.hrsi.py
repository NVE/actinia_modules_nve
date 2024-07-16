#!/usr/bin/env python3

"""
 MODULE:       t.import.hrsi
 AUTHOR(S):    Stefan Blumentrath
 PURPOSE:      Download and import Copernicus High Resolution Snow and Ice (HRSI)
               Monitoring data as a Space Time Raster Dataset (STRDS)
 COPYRIGHT:    (C) 2023 by stefan.blumentrath, and the GRASS Development Team

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
# % description: Download and import Copernicus High Resolution Snow and Ice Monitoring data as a Space Time Raster Dataset
# % keyword: temporal
# % keyword: download
# % keyword: import
# % keyword: raster
# % keyword: time
# % keyword: copernicus
# % keyword: cryosphere
# % keyword: snow
# % keyword: ice
# %end

# %flag
# % key: e
# % description: Extend existing STRDS (requires overwrite flag)
# % guisection: Settings
# %end

# %flag
# % key: l
# % description: Link the raster files using r.external
# % guisection: Settings
# %end

# %flag
# % key: f
# % description: Link the raster files in a fast way, without reading metadata using r.external
# % guisection: Settings
# %end

# %flag
# % key: p
# % description: Print info on search results (no download)
# % guisection: Settings
# %end

# %flag
# % key: g
# % description:  Print info on search results in shell script style (no download)
# % guisection: Settings
# %end

# %flag
# % key: m
# % description: Start and end time are publication time (modification)
# % guisection: Filter
# %end

# %flag
# % key: o
# % description: Override projection check
# % guisection: Output
# %end

# %flag
# % key: w
# % description: Write a log-file to the output directory
# % guisection: Output
# %end

# %option G_OPT_STRDS_OUTPUT
# % required: no
# % multiple: no
# % description: Name of the output space time raster dataset
# %end

# %option
# % key: output_directory
# % required: no
# % multiple: no
# % answer: ./
# % description: Path to the output directory where downloaded raw data are stored (default is the current directory)
# %end

# %option
# % key: product_type
# % type: string
# % required: yes
# % multiple: no
# % options: FractionalSnowCover,GapfilledFractionalSnowCover,PersistentSnowArea,PersistentSnowArea_LAEA,RiverandLakeIceExtent_S1,RiverandLakeIceExtent_S2,RiverandLakeIceExtent_S1_S2,AggregatedRiverandLakeIceExtent,SARWetSnow,WetDrySnow
# % key_desc: Product type to search and download
# % description: Product type to search and download
# % guisection: Filter
# %end

# %option
# % key: cloud_cover
# % type: integer
# % required: no
# % multiple: no
# % key_desc: memory in MB
# % label: Maximum cloud cover in products to download
# % description: Maximum cloud cover in products to download
# % guisection: Filter
# %end

# %option G_OPT_F_INPUT
# % key: aoi
# % type: string
# % required: no
# % multiple: no
# % key_desc: Input GeoJSON file with area of interest (AOI)
# % description: Input GeoJSON file with area of interest (AOI)
# % guisection: Filter
# %end

# %option
# % key: start_time
# % label: Earliest timestamp of temporal extent to include in the output
# % description: Timestamp in ISO format: "YYYY-MM-DD HH:MM:SS"
# % type: string
# % required: no
# % multiple: no
# % guisection: Filter
# %end

# %option
# % key: end_time
# % label: Latest timestamp of temporal extent to include in the output
# % description: Timestamp in ISO format: "YYYY-MM-DDTHH:MM:SS"
# % type: string
# % required: no
# % multiple: no
# % guisection: Filter
# %end

# %option
# % key: batch_size
# % label: Size of batches of files to download
# % type: integer
# % required: no
# % multiple: no
# % answer: 500
# % guisection: Filter
# %end

# %option G_OPT_F_INPUT
# % key: credits_file
# % type: string
# % required: no
# % multiple: no
# % guisection: Settings
# % key_desc: Input file with user credits for HRSI (can also be provided as environment variables)
# % description: Input file with user credits for HRSI (can also be provided as environment variables)
# %end

# %option
# % key: memory
# % type: integer
# % required: no
# % multiple: no
# % key_desc: memory in MB
# % label: Maximum memory to be used (in MB)
# % description: Cache size for raster rows
# % answer: 300
# % guisection: Settings
# %end

# %option
# % key: nprocs
# % type: integer
# % required: no
# % multiple: no
# % key_desc: Number of cores
# % label: Number of cores to use during import
# % answer: 1
# % guisection: Settings
# %end

# %rules
# % required: -g,-p,output
# % excludes: output,-p,-g
# %end

import os
import json
import re
import sys
import zipfile

from copy import deepcopy
from datetime import datetime
from http.client import IncompleteRead
from io import BytesIO

# from multiprocessing import Pool
from concurrent import futures
from pathlib import Path
from subprocess import PIPE
from urllib import parse
from urllib import request

# Non-builtin imports
import numpy as np
import grass.script as gs


class CLCCryoDownloader:
    """
    Basic class for listing, downloading, extracting and parsing data from
    Copernicus Land Services Cryosphere

    https://cryo.land.copernicus.eu/resto/api/collections/HRSI/describe.xml

    :todo: a) option to only update if needed and also clean in future
           Globbing the download_dir could help
    """

    def __init__(
        self,
        description_url="https://cryo.land.copernicus.eu/resto/api/collections/HRSI/describe.xml",
        token_url="https://cryo.land.copernicus.eu/auth/realms/cryo/protocol/openid-connect/token",
        output_directory="./",
        gisenv=None,
        credits_file=None,
        import_module=None,
        category_module=None,
        batch_size="1000",
        cores=1,
    ):
        #: Number of cores to use for parallel download, extraction and import (if relevant)
        self.cores = cores
        #: Attribute containg the URL for description of the Copernicus Cryosphere data search and download API
        self.description_url = description_url
        #: Attribute containg the URL for getting anaccess token for the Copernicus Cryosphere data search and download API
        self.token_url = "https://cryo.land.copernicus.eu/auth/realms/cryo/protocol/openid-connect/token"
        #: Attribute containg the API description as a dictionary
        self.api_description_dict = {}
        self.__parse_api_description()
        #: Attribute containg the URL for retrieve an access token for downloading Copernicus Cryosphere data
        self.token_url = token_url
        #: Attribute for storing a dictionary with the access token for downloading Copernicus Cryosphere data
        self.token = {}
        #: Attribute containg directory to which data is downloaded or written to
        self.output_directory = output_directory
        # Check if download directory is writable
        check_permissions(self.output_directory, "Download")
        #: Attribute containg a maximum number for retries for incomplete downloads (may happen if connection is closed prematurely or empty chunk of data is send
        self.max_retries = 10
        #: Attribute containg credits for cryo.land.copernicus.eu as tuple (username, password)
        self.user_credits = (None, None)
        self.__get_cryo_land_credits(credits_file)
        #: Attribute containing the search url
        self.search_url = None
        #: Attribute containing initial search results
        self.initial_search_result = None
        #: Attribute containing a dictionary describing the requested HRSI product
        self.requested_hrsi_product = {}
        #: Attribute containing path to a temporary file used for registering raster maps
        self.tempfile = gs.tempfile(create=True)
        #: Attribute containg the query time
        self.query_time = None
        #: Attribute to store the GRASS GIS module object to be used for import
        self.import_module = import_module
        #: Attribute to store the GRASS GIS module object to be used for adding category labels
        self.category_module = category_module
        #: Attribute storing the GRASS GIS environment variables
        self.gisenv = gisenv or dict(gs.gisenv())
        self.gisenv["LOCATION_PROJECTION"] = gs.read_command(
            "g.proj", flags="wf"
        ).strip()
        #: Attribute containing the projection information of the GRASS GIS location
        self.reference_crs = gs.read_command("g.proj", flags="wf").strip()
        #: Attribute containing the path to the directory where VRT files are stored
        self.vrt_dir = str(
            Path(self.gisenv["GISDBASE"]).joinpath(
                self.gisenv["LOCATION_NAME"], self.gisenv["MAPSET"], "gdal"
            )
        )
        Path(self.vrt_dir).mkdir(parents=True, exist_ok=True)
        #: Attribute defining if existing data should be overwritten
        self.recreate = gs.overwrite()
        self.batch_size = batch_size

    def __parse_api_description(self, api_format="application/json"):
        """Extracts supported search parameters from API description XML
        Here the JSON API format is used
        Only two hirarchy levels are assumed in the API description"""
        xml_description = request.urlopen(self.description_url)
        root = etree.fromstring(xml_description.read())  # noqa: S320

        desciption_dict = {}
        for item in root.iterchildren():
            children = item.getchildren()
            if not children:
                desciption_dict[str(item.tag.split("}")[1])] = item.text
            else:
                if (
                    item.tag.endswith("Url")
                    and "type" in item.attrib
                    and item.attrib["type"] == api_format
                ):
                    desciption_dict["API"] = dict(item.attrib)

                    desciption_dict["API"]["parameters"] = {}
                    for child in children:
                        desciption_dict["API"]["parameters"][child.attrib["name"]] = {
                            element[0]: element[1]
                            for element in child.items()
                            if element[0] != "name"
                        }

        self.api_description_dict = desciption_dict

    def __get_token(self):
        """Get Authentication token for download"""
        user, password = self.user_credits
        if not user:
            gs.fatal(_("Username is missing from input"))

        if not password:
            gs.fatal(_("Password is missing from input"))

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"
        }

        data = parse.urlencode(
            {
                "client_id": "PUBLIC",
                "grant_type": "password",
                "username": user,
                "password": password,
            }
        ).encode()

        req = request.Request(self.token_url, headers=headers, data=data)
        with request.urlopen(req) as resp:
            token = resp.read()
            if not token:
                gs.fatal(
                    _(
                        "Authentification error, please check your credentials for https://cryo.land.copernicus.eu/finder/."
                    )
                )

        self.token = json.loads(token)

    def __construct_search_url(
        self,
        query_params=None,
    ):
        """Construct HRSI search URL"""
        url_parts = parse.urlparse(self.api_description_dict["API"]["template"])
        query_parts = parse.parse_qs(url_parts.query)
        # Check query parameters:
        for query_param in query_params:
            if query_param not in query_parts:
                gs.fatal(
                    _(
                        "Invalid query parameter {0}. Only the following parameters are supported: {1}"
                    ).format(query_param, ", ".join(query_parts.keys()))
                )
        # Replace template with actual query
        url_parts = url_parts._replace(
            query="&".join(
                ["=".join(query_param) for query_param in query_params.items()]
            )
        )
        # Attach query URL to object
        self.search_url = url_parts.geturl()

    def initialize_search(self, query_params, hrsi_product):
        """Get results of an initial search"""
        self.__construct_search_url(query_params)
        with request.urlopen(self.search_url) as req:
            resp = json.loads(req.read())
            if resp["properties"]["totalResults"] == 0:
                gs.warning(_("Search returned no results"))
                sys.exit()
        self.initial_search_result = resp
        self.requested_hrsi_product = hrsi_product

    def print_search_info(self, shell_script_style):
        """Method to print search results"""
        if not shell_script_style:
            print(json.dumps(self.initial_search_result["properties"]))
        else:
            for key, val in self.initial_search_result["properties"].items:
                print(f"{key}={val}")
        sys.exit()

    def fetch_data(self, query_params, product_metadata):
        """Wrapper method to execute download in batches"""
        check_permissions(self.output_directory, "Download")
        # Minimize pageing
        query_params["maxRecords"] = self.batch_size
        batches_n = int(
            np.ceil(
                self.initial_search_result["properties"]["totalResults"]
                / float(self.batch_size)
            )
        )
        batch_number = 0
        gs.verbose(
            _(
                "Downloading a total of {files} files in {batches} batches of data"
            ).format(
                files=self.initial_search_result["properties"]["totalResults"],
                batches=batches_n,
            )
        )
        self.__construct_search_url(query_params)
        self.requested_hrsi_product = product_metadata
        next_batch = True
        url = self.search_url
        while next_batch:
            batch_number = batch_number + 1
            self.__get_token()
            gs.verbose(
                _("Downloading batch {n}: {url}").format(n=batch_number, url=url)
            )
            with request.urlopen(url) as req:
                resp = req.read()
                resp_dict = json.loads(resp)
            download_urls = [
                f["properties"]["services"]["download"]["url"]
                for f in resp_dict["features"]
            ]

            files_n = len(download_urls)
            active_cores = min(files_n, self.cores)
            if active_cores == 1:
                result_dicts = [self._download_and_import_data(download_urls)]
            elif active_cores > 1:
                # with Pool(active_cores) as pool:
                with futures.ThreadPoolExecutor(active_cores) as pool:
                    result_dicts = pool.map(
                        self._download_and_import_data,
                        [
                            download_urls[worker :: self.cores]
                            for worker in range(self.cores)
                        ],
                    )
            else:
                gs.warning(_("Nothing to download in current datasets"))
                sys.exit()

            next_page = [
                link
                for link in resp_dict["properties"]["links"]
                if "rel" in link and link["rel"] == "next"
            ]
            if next_page:
                url = next_page[0]["href"]
            else:
                next_batch = False

            with open(self.tempfile, "a", encoding="UTF8") as g_tempfile:
                g_tempfile.write(
                    "".join(
                        [
                            result_dict["register_strings"]
                            for result_dict in result_dicts
                            if result_dict["register_strings"]
                        ]
                    )
                )
            gs.percent(batch_number, batches_n, 1)

    def __get_cryo_land_credits(self, credits_file=None):
        """
        Method to fetch user credits for cryo.land.copernicus.eu and store it in the
        :attr:`~CLCCryoDownloader.user_credits` attribute.
        Credentials for authentication to cryo.land.copernicus.eu API can be either given
        using a credtis_file (see .cryo_land.example) or by defining
        environment variables:

        :envvar:`HRSI_USERNAME`
        :envvar:`HRSI_PASSWORD`

        The user's :envvar:`HOME` directoy is allways searched for a .cryo_land
        credits file.

        :param credits_file: Path to credits file to read
        :type credits_file: str
        """
        # Get authentication
        user = os.environ.get("HRSI_USERNAME")
        password = os.environ.get("HRSI_PASSWORD")

        credits_file = Path(credits_file or os.path.expanduser("~/.cryo_land"))
        if credits_file.exists():
            try:
                user, password = credits_file.strip().read_text().split("\n", 1)
            except OSError:
                gs.fatal(_("Unable to get credentials from credentials file {}").format(str(credits_file)))
        if not all((user, password)):
            gs.warning(
                _(
                    "No authentication provided. Downloading data is thus not possible.\n"
                    "Please provide authentication information"
                )
            )
        self.user_credits = (user, password)

    def _download_and_import_data(self, download_urls):
        """
        Private method to download all data in the the atom_dict from geonorge nedlasting API

        :todo: Should only update if needed and also clean in future
               Globbing the download_dir could help

        """
        register_strings = []
        failed_downloads = []
        # Download dict content
        for download_url in download_urls:
            # gs.verbose(_("Downloading {}...").format(download_url))
            # Initialize request
            hrsi_request = request.Request(download_url)
            # Add token authorization
            hrsi_request.add_header(
                "Authorization",
                f"{self.token['token_type']} {self.token['access_token']}",
            )

            # Download data
            retries = 0
            success = False
            data = BytesIO()
            while not success and retries < self.max_retries:
                try:
                    with request.urlopen(hrsi_request) as response:
                        data.write(response.read())
                    success = True
                except IncompleteRead as partial_read:
                    data.write(partial_read.partial)
                    hrsi_request.add_header("Range", f"bytes={data.tell()}-")
                    retries += 1
            if not success:
                gs.warning(_("Error when downloading {}").format(download_url))
                failed_downloads.append(download_url)

            # Extract metadata
            with zipfile.ZipFile(data) as zip_file:
                hrsi_file_xml = [
                    hrsi_file
                    for hrsi_file in zip_file.namelist()
                    if hrsi_file.endswith("xml")
                ][0]
                zip_data = zip_file.read(hrsi_file_xml)
                hrsi_file_path = Path(self.output_directory) / Path(hrsi_file_xml).name
                hrsi_file_path.write_bytes(zip_data)

                # temporal extend is not consistently represented in the
                # metadata, so this part of the code is deactivated
                if not self.requested_hrsi_product["time_pattern"]:
                    metadata_xml = etree.fromstring(zip_data)  # noqa: S320
                    meta_data = MD_Metadata(metadata_xml)

                    # title = meta_data.identification[0].title
                    # description = meta_data.identification[0].abstract
                    start = meta_data.identification[0].temporalextent_start
                    end = meta_data.identification[0].temporalextent_end
                    if start == end:
                        end = None

                # Extract data
                first_tif = 0
                for hrsi_file in zip_file.namelist():
                    # XML file is already extracted
                    if hrsi_file.endswith("xml"):
                        continue

                    zip_data = zip_file.read(hrsi_file)
                    hrsi_file_path = Path(self.output_directory) / Path(hrsi_file).name
                    hrsi_file_path.write_bytes(zip_data)
                    map_name = legalize_name_string(hrsi_file_path.stem)
                    full_map_name = f"{map_name}@{self.gisenv['MAPSET']}"
                    sub_product = map_name.split("_")[-1]

                    gdal_dataset = gdal.Open(str(hrsi_file_path))
                    if first_tif == 0:
                        equal_projection, transform = check_projection_match(
                            self.reference_crs, gdal_dataset
                        )
                        first_tif += 1
                        if self.requested_hrsi_product["time_pattern"]:
                            start = timestamp_from_filename(
                                hrsi_file, **self.requested_hrsi_product["time_pattern"]
                            ).strftime("%Y-%m-%d %H:%M:%S")
                            end = None
                    if not equal_projection:
                        # Create VRT
                        try:
                            input_path = create_vrt(
                                hrsi_file_path,
                                gdal_dataset,
                                self.gisenv,
                                equal_projection,
                                transform,
                                self.requested_hrsi_product[sub_product]["resample"],
                                self.requested_hrsi_product[sub_product]["data_type"],
                                data_range=self.requested_hrsi_product[sub_product][
                                    "valid_range"
                                ]
                                if self.import_module.flags.get("m")
                                else None,
                                nodata=self.requested_hrsi_product[sub_product][
                                    "nodata"
                                ],
                                recreate=self.recreate,
                            )
                            input_path = Path(input_path)
                        except Exception:
                            gdal_dataset = None
                            continue
                    else:
                        input_path = hrsi_file_path
                    gdal_dataset = None

                    # Link or import
                    import_mod = deepcopy(self.import_module)
                    import_mod.inputs.input = str(input_path)
                    import_mod.inputs.title = self.requested_hrsi_product[sub_product][
                        "title"
                    ]
                    import_mod.outputs.output = map_name
                    try:
                        import_mod.run()
                    except Exception:
                        gs.warning(
                            _("Could not import raster map {}").format(
                                str(hrsi_file_path)
                            )
                        )
                        hrsi_file_path.unlink()
                        continue

                    # Add categories if relevant
                    if self.requested_hrsi_product[sub_product]["categories"]:
                        Module(
                            "r.category",
                            rules="-",
                            map=map_name,
                            stdin="\n".join(
                                self.requested_hrsi_product[sub_product]["categories"]
                            ),
                            separator=":",
                            run_=True,
                        )

                    # Compile register string
                    register_strings.append(
                        "|".join(
                            [
                                full_map_name,
                                start,
                                end,
                                self.requested_hrsi_product[sub_product][
                                    "semantic_label"
                                ],
                            ]
                        )
                        + "\n"
                        if end
                        else "|".join(
                            [
                                map_name,
                                start,
                                self.requested_hrsi_product[sub_product][
                                    "semantic_label"
                                ],
                            ]
                        )
                        + "\n"
                    )

        return {
            "register_strings": "".join(register_strings),
            "failed_downloads": failed_downloads,
        }


def legalize_name_string(string):
    """Replace conflicting characters with _"""
    return re.sub(r"[^\w\d-]+|[^\x00-\x7F]+|[ -/\\]+", "_", string)


def timestamp_from_filename(
    file_path, pattern=None, pattern_prefix="_", pattern_suffix="_"
):
    """Extracts a timestamp as a datetime object from a file name using
    a pattern
    :param file_path: a pathlib Path object
    :param pattern: strftime formated sting describing the pattern of the
                    timestamp in the file name
    :return: datetime object of the time stamp extracted from file name
    :rtype: datetime
    """
    date_pattern = f".*{pattern_prefix}({strftime_to_regex(pattern)}){pattern_suffix}.*"
    time_string = re.match(date_pattern, str(file_path)).groups()[0]
    return datetime.strptime(time_string, pattern)


def strftime_to_regex(string):
    """
    Transform a strftime format string to a regex for extracting time from file name
    :param string: strftime format string
    :return: regex string
    """
    digits = {
        "%Y": "[0-9]{4}",
        "%m": "[0-9]{2}",
        "%d": "[0-9]{2}",
        "%M": "[0-9]{2}",
        "%H": "[0-9]{2}",
        "%S": "[0-9]{2}",
    }
    for item, digit in digits.items():
        string = string.replace(item, digit)
    if "%" in string:
        gs.warning(_("Unmatched format item in input string"))
    return string


def check_permissions(directory, mode):
    """Check if directory is writable"""
    if not os.access(str(directory), os.W_OK):
        gs.fatal(
            _("Cannot write to directory <{directory}>. {mode} will fail.").format(
                directory=str(directory), mode=mode
            )
        )


def create_vrt(
    input_dataset_path,
    input_dataset,
    gisenv,
    equal_proj,
    transform,
    resample,
    data_type,
    data_range=None,
    nodata=None,
    recreate=False,
):
    """Create a GDAL VRT for import"""
    vrt_dir = Path(gisenv["GISDBASE"]).joinpath(
        gisenv["LOCATION_NAME"], gisenv["MAPSET"], "gdal"
    )
    vrt_path = vrt_dir.joinpath(f"{input_dataset_path.stem}.vrt")
    if vrt_path.exists() and not recreate:
        return vrt_path
    kwargs = {
        "format": "VRT",
        "outputType": data_type,
    }
    if equal_proj:
        if nodata is not None:
            kwargs["noData"] = nodata
        vrt = gdal.Translate(
            vrt_path,
            input_dataset,  # Use already opened dataset here
            options=gdal.TranslateOptions(
                **kwargs,
                # stats=True,
                # outputBounds=
            ),
        )
    else:
        geo_transform = input_dataset.GetGeoTransform()
        x_res = np.abs(geo_transform[5])
        y_res = np.abs(geo_transform[5])
        transformed_bbox = transform_bounding_box(
            (
                geo_transform[0],
                geo_transform[3] + geo_transform[5] * input_dataset.RasterYSize,
                geo_transform[0] + geo_transform[1] * input_dataset.RasterXSize,
                geo_transform[3],
            ),
            transform,
            edge_densification=15,
        )
        # Alining to input resolution
        aligned_bbox = align_bbox(transformed_bbox, x_res, y_res)
        kwargs["dstSRS"] = gisenv["LOCATION_PROJECTION"]
        kwargs["resampleAlg"] = resample
        kwargs["xRes"] = x_res
        kwargs["yRes"] = y_res
        kwargs["outputBounds"] = (
            aligned_bbox["west"],
            aligned_bbox["south"],
            aligned_bbox["east"],
            aligned_bbox["north"],
        )

        if nodata is not None:
            kwargs["srcNodata"] = nodata
        vrt = gdal.Warp(
            str(vrt_path),
            input_dataset,
            options=gdal.WarpOptions(**kwargs),
        )
    # Define theoretical range
    if data_range:
        band = vrt.GetRasterBand(1)
        band.SetMetadataItem("STATISTICS_MINIMUM", str(data_range[0]))
        band.SetMetadataItem("STATISTICS_MAXIMUM", str(data_range[1]))
    vrt = None

    return str(vrt_path)


def align_bbox(bbox, resolution_x, resolution_y):
    """Align bounding box coordinates to match resolution"""
    bbox_dict = gs.parse_command(
        "g.region",
        flags="agu",
        n=bbox[3],
        e=bbox[2],
        w=bbox[0],
        s=bbox[1],
        ewres=resolution_x,
        nsres=resolution_y,
    )
    return {
        "west": bbox_dict["w"],
        "south": bbox_dict["s"],
        "east": bbox_dict["e"],
        "north": bbox_dict["n"],
    }


def check_projection_match(reference_crs, input_dataset):
    """Check if projections match with projection of the location
    using gdal/osr
    """
    input_dataset_crs = input_dataset.GetSpatialRef()
    location_crs = osr.SpatialReference()
    location_crs.ImportFromWkt(reference_crs)
    projections_match = input_dataset_crs.IsSame(location_crs)
    transform = None
    if not projections_match:
        transform = osr.CoordinateTransformation(input_dataset_crs, location_crs)

    return projections_match, transform


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
    return parse.quote(ogr_feature.geometry().ExportToIsoWkt())


def transform_bounding_box(bbox, transform, edge_densification=15):
    """Transform the datasets bounding box into the projection of the location
    with desified edges
    bbox is a tuple of (xmin, ymin, xmax, ymax)
    Adapted from:
    https://gis.stackexchange.com/questions/165020/how-to-calculate-the-bounding-box-in-projected-coordinates
    """
    u_l = np.array((bbox[0], bbox[3]))
    l_l = np.array((bbox[0], bbox[1]))
    l_r = np.array((bbox[2], bbox[1]))
    u_r = np.array((bbox[2], bbox[3]))

    def _transform_vertex(vertex):
        x_transformed, y_transformed, _ = transform.TransformPoint(*vertex)
        return (x_transformed, y_transformed)

    # This list comprehension iterates over each edge of the bounding box,
    # divides it into `edge_densification` number of points, then reduces
    # that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    return [
        bounding_fn(
            [
                _transform_vertex(p_a * v + p_b * (1 - v))
                for v in np.linspace(0, 1, edge_densification)
            ]
        )
        for p_a, p_b, bounding_fn in [
            (u_l, l_l, lambda point_list: min(p[0] for p in point_list)),
            (l_l, l_r, lambda point_list: min(p[1] for p in point_list)),
            (l_r, u_r, lambda point_list: max(p[0] for p in point_list)),
            (u_r, u_l, lambda point_list: max(p[1] for p in point_list)),
        ]
    ]


def main():
    """Do the main work"""
    # Define relevant HRSI product metadata
    hrsi_products = {
        "FractionalSnowCover": {
            "id": "FSC",
            "title": "Sentinel-2 Fractional Snow Cover (FSC) 20m",
            "description": "The Fractional Snow Cover (FSC) product is generated in NRT for the entire EEA38+UK based "
            "on optical satellite data from the S2 constellation. The product provides the fraction of the "
            "surface covered by snow at the top of canopy and on ground per pixel as a percentage (0% - "
            "100%) with a spatial resolution of 20 m x 20 m.",
            "resolution": 20,
            "mission": "S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "CLD": {
                    "title": "Presence of clouds and cloud shadows",
                    "description": "CLD layer indicates the presence of clouds and cloud shadows, and corresponds to the MAJA L2A cloud mask",
                    "resample": "near",
                    "semantic_label": "cloud_masks",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "FSCOG": {
                    "title": "On-ground fractional snow cover",
                    "description": "On-ground fractional snow cover (%) and associated information",
                    "resample": "bilinear",
                    "semantic_label": "fractional_snow_cover_on_ground",
                    "units": "percent",
                    "nodata": [205, 255],
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "FSCTOC": {
                    "title": "Top of canopy fractional snow cover",
                    "description": "Top of canopy fractional snow cover (%), and associated information",
                    "resample": "bilinear",
                    "semantic_label": "fractional_snow_cover_top_of_canopy",
                    "units": "percent",
                    "nodata": [205, 255],
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "NDSI": {
                    "title": "Normalised difference snow index (NDSI)",
                    "description": "Normalised difference snow index (%) of detected snow areas",
                    "resample": "bilinear",
                    "semantic_label": "S2_NDSI",
                    "units": "percent",
                    "nodata": [205, 255],
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCFLAGS": {
                    "title": "Expert flags providing quality information obtained during the processing of FSCTOC and FSCOG",
                    "description": "QCFLAGS are expert flags providing quality information obtained during the processing of FSCTOC and FSCOG",
                    "resample": "near",
                    "semantic_label": "S2_FSC_QCFLAGS",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCOG": {
                    "title": "Confidence level of the FSCOG layer",
                    "description": "QCOG indicates the confidence level of the FSCOG layer",
                    "resample": "near",
                    "semantic_label": "S2_FSC_QCOG",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCTOC": {
                    "title": "Confidence level of the FSCTOC layer",
                    "description": "QCTOC indicates the confidence level of the FSCTOC layer",
                    "resample": "near",
                    "semantic_label": "S2_FSC_QCTOC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "GapfilledFractionalSnowCover": {
            "id": "GFSC",
            "title": "Sentinel-1 + Sentinel-2 Daily cumulative Gap-filled Fractional Snow Cover (GFSC) 60m",
            "description": "The daily cumulative Gap-filled Fractional Snow Cover (GFSC) product is generated in NRT for "
            "the entire EEA38+UK domain based on SAR data from the S1 constellation and optical data "
            "from the S2 constellation. The product merges the latest observations available to form a "
            "spatially complete overview of snow conditions. The product provides the extent of the snow "
            "cover per pixel as a percentage (0% - 100%) with a spatial resolution of 60 m x 60 m. The "
            "product uses FSC, WDS and SWS products as input to form a spatially complete composite of "
            "snow conditions, to reduce observational gaps due to clouds and lack of sensor coverage on a "
            "daily basis. The product applies the on-ground FSC (FSCOG) and SWS and "
            "presents the combined information as FSC.",
            "resolution": 60,
            "mission": "S1-S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%d", "pattern_suffix": "-"},
                "AT": {
                    "title": "Sensing start date, seconds from Unix time",
                    "description": "Sensing start date, seconds from Unix time",
                    "resample": "near",
                    "semantic_label": "sensing_time",
                    "units": "seconds from Unix time",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 4294967295],
                    "data_type": gdal.GDT_UInt32,
                },
                "GF": {
                    "title": "Fractional snow cover (%) and associated information",
                    "description": "Fractional snow cover (%) and associated information",
                    "resample": "bilinear",
                    "semantic_label": "fractional_snow_cover",
                    "units": "percent",
                    "nodata": [205, 255],
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Quality layer providing basic assessment of FSC, WDS or SWS quality",
                    "description": "Quality layer providing basic assessment of FSC, WDS or SWS quality",
                    "resample": "near",
                    "semantic_label": "S1_S2_FSC_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCFLAGS": {
                    "title": "Expert quality flags related to GFSC product quality",
                    "description": "Expert quality flags related to GFSC product quality",
                    "resample": "near",
                    "semantic_label": "S1_S2_FSC_QCFLAGS",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "PersistentSnowArea": {
            "id": "PSA_WGS84",
            "title": "Sentinel-2 Persistent Snow Area (PSA) 20m",
            "description": "The PSA product is an annual product that is derived from the top of canopy Fractional Snow "
            "Cover (FSCTOC) product for the entire EEA38+UK. We refer users to section 2 regarding the "
            "description of the algorithm and the content of FSC products. Each PSA product gives access to "
            "the persistent snow cover during a particular hydrological year at the resolution of 20 m x 20 m.",
            "resolution": 20,
            "mission": "S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": None,
                "PSA": {
                    "title": "Persistent snow area",
                    "description": "Persistent snow area",
                    "resample": "near",
                    "semantic_label": "S2_PSA",
                    "units": "bool",
                    "nodata": [255],
                    "categories": [
                        "0:no persistent snow",
                        "1:persistent snow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Confidence level of the PSA layer",
                    "description": "Quality layer providing basic assessment of PSA quality (based on the number of clear-sky observations)",
                    "resample": "near",
                    "semantic_label": "S2_PSA_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "PersistentSnowArea_LAEA": {
            "id": "PSA_LAEA",
            "title": "Sentinel-2 Persistent Snow Area (PSA) 20m",
            "description": "The PSA product is an annual product that is derived from the top of canopy Fractional Snow "
            "Cover (FSCTOC) product for the entire EEA38+UK. We refer users to section 2 regarding the "
            "description of the algorithm and the content of FSC products. Each PSA product gives access to "
            "the persistent snow cover during a particular hydrological year at the resolution of 20 m x 20 m.",
            "resolution": 20,
            "mission": "S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": None,
                "PSA": {
                    "title": "Persistent snow area",
                    "description": "Persistent snow area",
                    "resample": "near",
                    "semantic_label": "S2_PSA",
                    "units": "bool",
                    "nodata": [255],
                    "categories": [
                        "0:no persistent snow",
                        "1:persistent snow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Confidence level of the PSA layer",
                    "description": "Quality layer providing basic assessment of PSA quality (based on the number of clear-sky observations)",
                    "resample": "near",
                    "semantic_label": "S2_PSA_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "RiverandLakeIceExtent_S1": {
            "id": "RLIE",
            "title": "Sentinel-1 River and Lake Ice Extent S1 (RLIE S1) 20m",
            "description": "The S1 River and Lake Ice Extent (RLIE S1) product is generated in NRT for the entire "
            "EEA38+UK based on synthetic aperture radar data from the S1 constellation. The product "
            "focuses on the surface water areas defined by the EU-Hydro database and provides "
            "information about river and lake areas covered by snow-covered or snow-free ice, at a spatial "
            "resolution of 20 m x 20 m. For the sake of consistency across RLIE products, the RLIE S1 "
            "product is delivered on the S2 Level-1C tiling grid, with a pixel size of 20 m x 20 m.",
            "resolution": 20,
            "mission": "S1",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "RLIE": {
                    "title": "Sentinel-1 River and Lake Ice Extent (RLIE S1)",
                    "description": "Sentinel-1 River and Lake Ice Extent (RLIE S1)",
                    "resample": "near",
                    "semantic_label": "S1_RLIE",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "1:open water",
                        "100:snow-covered or snow-free ice",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Quality layer providing basic assessment of RLIE S1 quality",
                    "description": "Quality layer providing basic assessment of RLIE S1 quality",
                    "resample": "near",
                    "semantic_label": "S1_RLIE_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCFLAGS": {
                    "title": "Quality flags related to RLIE product quality",
                    "description": "Quality flags related to RLIE product quality",
                    "resample": "near",
                    "semantic_label": "S1_RLIE_QCFLAGS",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "RiverandLakeIceExtent_S2": {
            "id": "RLIE",
            "title": "Sentinel-2 River and Lake Ice Extent (RLIE S2) 20m",
            "description": "The S2 River and Lake Ice Extent (RLIE S2) product is generated in NRT for the entire "
            "EEA38+UK based on optical satellite data from the S2 constellation. The product focuses on "
            "the surface water areas defined by the EU-Hydro database and provides the river and "
            "lake area covered by snow-covered or snow-free ice, at a spatial resolution of 20 m x 20 m.",
            "resolution": 20,
            "mission": "S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "RLIE": {
                    "title": "Sentinel-2 River and Lake Ice Extent (RLIE S2)",
                    "description": "Sentinel-2 River and Lake Ice Extent (RLIE S2)",
                    "resample": "near",
                    "semantic_label": "S2_RLIE",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "1:open water",
                        "100:snow-covered or snow-free ice",
                        "254:other features",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Quality layer providing basic assessment of RLIE S2 quality",
                    "description": "Quality layer providing basic assessment of RLIE S2 quality",
                    "resample": "near",
                    "semantic_label": "S2_RLIE_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCFLAGS": {
                    "title": "Quality flags related to RLIE product quality",
                    "description": "Quality flags related to RLIE product quality",
                    "resample": "near",
                    "semantic_label": "S2_RLIE_QCFLAGS",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "RiverandLakeIceExtent_S1_S2": {
            "id": "RLIE",
            "title": "Sentinel-1 + Sentinel-2 River and Lake Ice Extent S1+S2 (RLIE S1+S2) 20m",
            "description": "S1 and S2 River and Lake Ice Extent (RLIE S1+S2) is a product generated in delayed-time for "
            "the entire EEA38+UK according to RLIE S1 and RLIE S2 overlap. The RLIE S1+S2 is computed "
            "as a combination of RLIE S1 and RLIE S2 products acquired on the same date. The product "
            "focuses on the surface water areas defined by the EU-Hydro database and provides "
            "river and lake areas covered by ice, at a spatial resolution of 20 m x 20 m on the S2 tiling grid.",
            "resolution": 20,
            "mission": "S1-S2",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "RLIE": {
                    "title": "Sentinel-1 + Sentinel-2 River and Lake Ice Extent (RLIE S1+S2)",
                    "description": "Sentinel-1 + Sentinel-2 River and Lake Ice Extent (RLIE S1+S2)",
                    "resample": "near",
                    "semantic_label": "S1_S2_RLIE",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "1:open water",
                        "100:snow-covered or snow-free ice",
                        "254:other features",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QC": {
                    "title": "Quality layer providing basic assessment of RLIE S1+S2 quality",
                    "description": "Quality layer providing basic assessment of RLIE S1+S2 quality",
                    "resample": "near",
                    "semantic_label": "S1_S2_RLIE_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "205:cloud or cloud shadow",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCFLAGS": {
                    "title": "Quality flags related to RLIE product quality",
                    "description": "Quality flags related to RLIE product quality",
                    "resample": "near",
                    "semantic_label": "S1_S2_RLIE_QCFLAGS",
                    "units": "bits",
                    "nodata": None,
                    "categories": None,
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        # ARLIE is vector data and currently not supported
        "AggregatedRiverandLakeIceExtent": {
            "id": "ARLIE",
            "title": "",
            "description": "",
            "resolution": 20,
            "mission": "S1-S2",
            "suffix": ".gdb.zip",
            "subdatasets": ["ARLIE"],
        },
        "SARWetSnow": {
            "id": "SWS",
            "title": "Sentinel-1 + Sentinel-2 SAR Wet Snow in high mountains (SWS) 60m",
            "description": "The S1 based SAR Wet Snow (SWS) classification is applicable for high-mountain areas within"
            'selected S2 tiles of the EEA38+UK domain. For the SWS product generation, "high mountains" '
            "means areas above a certain elevation where no human activities such as tilling agricultural "
            "areas are affecting the SAR signal. The SWS product is generated in NRT based on C-band SAR "
            "satellite data from the S1 constellation. The product provides binary information on the wet "
            "snow extent and the snow free or patchy snow or dry snow extent in high mountain areas. "
            "The SWS product is provided with a grid spacing of 60 m by 60 m.",
            "resolution": 60,
            "mission": "S1",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "WSM": {
                    "title": "Wet Snow classification in high Mountains areas (WSM)",
                    "description": "Wet Snow classification in high Mountains areas (WSM) and associated information",
                    "resample": "near",
                    "semantic_label": "wet_snow_classes",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "110:Wet snow",
                        "125:Dry snow or snow free or patchy snow",
                        "200:Radar Shadow / layover / foreshortening (masked)",
                        "210:Water (masked, from EU-Hydro)",
                        "220:Forest (masked, from Copernicus HRL TCD, 2015)",
                        "230:Urban area (masked, from Copernicus HRL IMD, 2018)",
                        "240:Non-mountain areas (masked, from Copernicus DEM GLO30 and CORINE Land Cover 2018)",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCWSM": {
                    "title": "Quality layer providing basic assessment of WSM quality",
                    "description": "Quality layer providing basic assessment of WSM quality",
                    "resample": "near",
                    "semantic_label": "wet_snow_classes_quality",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "250:masked",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
        "WetDrySnow": {
            "id": "WDS",
            "title": "Sentine-1 + Sentinel-2 Wet/Dry Snow (WDS) 60m",
            "description": "The Wet / Dry Snow (WDS) product provides information on the snow state (wet or dry) by "
            "combining S1 based wet snow maps within the snow cover extent observed by means of S2 "
            "data (cf. Section 2). The WDS product is provided for each S2 tile with a grid spacing of "
            "60 m by 60 m.",
            "resolution": 60,
            "mission": "S1",
            "suffix": ".tif",
            "subdatasets": {
                "time_pattern": {"pattern": "%Y%m%dT%H%M%S"},
                "SSC": {
                    "title": "Snow State Classification (SSC)",
                    "description": "Snow State Classification (SSC) within S2 FSCTOCagg product and associated information",
                    "resample": "near",
                    "semantic_label": "S2_SSC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "110:Wet snow (from S1 and FSCTOCagg >= THR)",
                        "115:Dry snow (masked, FSCTOCagg >= THR, no wet snow from S1)",
                        "120:Snowfree or patchy snow cover (masked, 0 <= FSCTOCagg < THR)",
                        "200:Radar shadow / layover / foreshortening (masked)",
                        "205:Cloud or cloud shadow (masked, from FSCTOCagg = cloud)",
                        "210:Water (masked, from EU-Hydro)",
                        "220:Forest (masked, from Copernicus HRL TCD, 2015)",
                        "230:Urban area (masked, from Copernicus HRL IMD, 2018)",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
                "QCSSC": {
                    "title": "Quality layer providing basic assessment of the wet snow state classification",
                    "description": "Quality layer providing basic assessment of the wet snow state classification",
                    "resample": "near",
                    "semantic_label": "S2_SSC_QC",
                    "units": "classes",
                    "nodata": [255],
                    "categories": [
                        "0:high quality",
                        "1:medium quality",
                        "2:low quality",
                        "3:minimal quality",
                        "250:masked",
                        "255:no data",
                    ],
                    "valid_range": [0, 255],
                    "data_type": gdal.GDT_Byte,
                },
            },
        },
    }

    # Check batch_size input
    if not 0 < int(options["batch_size"]) <= 2000:
        gs.fatal(_("Invalid input for batch_size. Valid range is 1-2000."))

    # Get GRASS GIS environment
    gisenv = dict(gs.gisenv())

    # Initialize TGIS
    tgis.init()

    # Initialize SpaceTimeRasterDataset (STRDS) using tgis
    strds_long_name = f"{options['output']}@{gisenv['MAPSET']}"
    tgis_strds = tgis.SpaceTimeRasterDataset(strds_long_name)

    # Check if target STRDS exists and create it if not or abort if overwriting is not allowed
    if tgis_strds.is_in_db() and not gs.overwrite():
        gs.fatal(
            _(
                "Output STRDS <{}> exists."
                "Use --overwrite with or without -e to modify the existing STRDS."
            ).format(options["output"])
        )

    # Setup module objects for import
    imp_flags = "o" if flags["o"] else ""
    if flags["l"] or flags["f"]:
        import_module = Module(
            "r.external",
            stdout_=PIPE,
            stderr_=PIPE,
            quiet=True,
            overwrite=gs.overwrite(),
            run_=False,
            flags=imp_flags + "ra" if flags["f"] else imp_flags + "ma",
        )
    else:
        import_module = Module(
            "r.in.gdal",
            stdout_=PIPE,
            stderr_=PIPE,
            quiet=True,
            overwrite=gs.overwrite(),
            run_=False,
            flags=imp_flags + "a",
            memory=options["memory"],
        )

    category_module = Module(
        "r.category",
        quiet=True,
        run_=False,
        separator=":",
        rules="-",
    )

    # Compile query parameters
    # User input may be validated with regex
    # Topic
    query_params = {
        "productType": hrsi_products[options["product_type"]]["id"],
        "mission": hrsi_products[options["product_type"]]["mission"],
    }

    # Spatial
    if not options["aoi"]:
        reg = Region()
        query_params["box"] = ",".join(
            [str(reg.west), str(reg.south), str(reg.east), str(reg.north)]
        )
    else:
        # GeoJSON input may be validated
        query_params["geometry"] = get_aoi_wkt(options["aoi"])

    run_time = datetime.now()
    # Temporal
    if options["start_time"]:
        if flags["m"]:
            query_params["publishedAfter"] = options["start_time"]
        else:
            query_params["startDate"] = options["start_time"]

    if options["end_time"]:
        if flags["m"]:
            query_params["publishedBefore"] = options["end_time"]
        else:
            query_params["completionDate"] = options["end_time"]
            # Make sure the query result does not change during download
            query_params["publishedBefore"] = run_time.isoformat()

    query_params["maxRecords"] = options["batch_size"]

    # Create download object
    clc_downloader = CLCCryoDownloader(
        output_directory=options["output_directory"],
        credits_file=options["credits_file"],
        import_module=import_module,
        category_module=category_module,
        batch_size=options["batch_size"],
        cores=int(options["nprocs"]),
    )

    # Initialize query
    clc_downloader.initialize_search(
        query_params, hrsi_products[options["product_type"]]["subdatasets"]
    )

    if flags["g"] or flags["p"]:
        clc_downloader.print_search_info(flags["g"])

    # Download data
    clc_downloader.fetch_data(
        query_params, hrsi_products[options["product_type"]]["subdatasets"]
    )

    # Create STRDS if needed
    if not tgis_strds.is_in_db() or (gs.overwrite() and not flags["e"]):
        Module(
            "t.create",
            output=options["output"],
            type="strds",
            temporaltype="absolute",
            title=hrsi_products[options["product_type"]]["title"],
            description=hrsi_products[options["product_type"]]["description"],
            verbose=True,
        )

    # Remove potential duplicates
    reg_lines = set(Path(clc_downloader.tempfile).read_text(encoding="UTF8").strip().split("\n"))

    # Write registration file with unique lines
    Path(clc_downloader.tempfile).write_text("\n".join(reg_lines) + "\n", encoding="UTF8")

    # Register downloaded maps in STRDS
    register_maps_in_space_time_dataset(
        "raster",
        strds_long_name,
        file=clc_downloader.tempfile,
        update_cmd_list=False,
        fs="|",
    )

    # Update mode and new download mode
    # Write log
    if flags["w"]:
        Path(clc_downloader.output_directory / f"hrsi_import_{run_time.strftime('%Y%m%dT%H%M%S')}.log", encoding="UTF8").write_text(json.dumps({"query_time": run_time.isoformat(), "query_params": query_params}, indent=2))


if __name__ == "__main__":
    options, flags = gs.parser()

    # Lazy imports
    import grass.temporal as tgis
    from grass.pygrass.gis.region import Region
    from grass.pygrass.modules.interface import Module
    from grass.temporal.register import register_maps_in_space_time_dataset

    try:
        from osgeo import gdal
        from osgeo import osr
        from osgeo import ogr
    except ImportError as import_error:
        gs.fatal(_("Module requires GDAL python bindings: {}").format(import_error))

    try:
        # from owslib.iso import etree
        from owslib.iso import MD_Metadata
    except ImportError as e:
        gs.fatal(_("Module requires owslib python library: {}").format(e))

    try:
        from lxml import etree
    except ImportError as import_error:
        gs.fatal(_("Module requires lxml python library: {}").format(import_error))

    sys.exit(main())
