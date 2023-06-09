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
# COPYRIGHT:	(C) 2020-2022 by mundialis GmbH & Co. KG and the GRASS Development Team
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
# % description: Path to aoi GeoJSON file
# % label: Path to aoi GeoJSON file
# %end

# %option
# % key: token
# % type: string
# % required: yes
# % multiple: no
# % description: Path to ASF token file
# % label: File has to contain ASF token
# %end

# %option
# % key: start
# % required: no
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
# % key: beamMode
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
# % required: start
# % collective: start,end
# %end

import zipfile
import os
import json
import hashlib
import concurrent.futures

from pathlib import Path

import shapely.wkt
import asf_search as asf
import geopandas as gpd

import grass.script as gs

def checksum_test(checksum, expected_checksum, _dfile):
    if checksum != expected_checksum:
        gs.verbose(f'Checksum test failed for {os.path.split(_dfile)[1]}')
        os.remove(_dfile)
        gs.verbose(f'{os.path.split(_dfile)[1]} is deleted')
        return False
    else:
        gs.verbose(f'Checksum test OK for {os.path.split(_dfile)[1]}')
        return True
    

def main():
    aoi = shapely.wkt.dumps(gpd.read_file(options["aoi"]).iloc[0].geometry)

    opts = {
            'platform': options["platform"],
            'beamMode': options["beamMode"],
            'processingLevel': options["processinglevel"],
            'start': options["start"],
            'end': options["end"],
            'intersectsWith': aoi
        }

    results = asf.geo_search(**opts)
    gs.verbose(f'{len(results)} results found')
    
    gs.verbose(f'Listing all scenes found:')
    for count, r in enumerate(results, start=1):
        gs.verbose(f'{r.properties['fileName']}')
    
    if flags["l"]:
        gs.verbose(f'All available scenes are listed.')
        sys.exit()
    
    token_file = Path(options["token"])

    with open(token_file, 'r') as f:
        token = f.read()
        print(token)
        f.close()
        
    token_session = asf.ASFSession().auth_with_token(token)

    download_dir = Path(options["output_directory"]).mkdir(exist_ok=True, parents=True)
    download_dir = Path(options["output_directory"])
    gs.verbose(f'Downloading to {download_dir}')
    
    for count, r in enumerate(results, start=1):
        _dfile = download_dir / r.properties['fileName']
        
        r.download(path=download_dir, session=token_session)  
        gs.verbose(f'{r.properties['fileName']} is downloaded')
        
        if flags['c']:
            i = 0
            while  i < 3:
                expected_checksum = r.properties['md5sum']
                checksum = hashlib.md5(_dfile.read_bytes()).hexdigest()
                
                if not checksum_test(checksum, expected_checksum, _dfile):
                    gs.verbose(f'Trying to download {os.path.split(_dfile)[1]} again')
                    r.download(path=download_dir, session=token_session)
                    gs.verbose(f'{r.properties['fileName']} is downloaded again')
                    i += 1
                else:
                    i = 3
           
                

if __name__ == "__main__":
    options, flags = grass.parser()
    main()
    




