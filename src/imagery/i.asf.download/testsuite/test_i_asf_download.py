"""Test i.asf.download for Sentinel-1

(C) 2023 by NVE, Yngve Antonsen and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Yngve Antonsen
"""

import os

import grass.script as gs

from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.tempdir = gs.tempdir()

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        gs.utils.try_rmdir(cls.tempdir)

    def test_asf_download(self):
        """Test download and import of Sentinel-1"""
        self.assertModule(
            "i.asf.download",
            platform="Sentinel-1",
            beam_mode="IW",
            processinglevel="GRD_HD",
            start="2023-03-15T00:00:00Z",
            end="2023-03-16T00:00:00Z",
            aoi="./data/roi.geojson",
            output_directory=self.tempdir,
            verbose=True,
        )

    def test_asf_parallel_download_with_checks(self):
        """Test parallel download and import of Sentinel-1 with
        checksum-test. Tests also if already downloaded files
        are skipped"""
        self.assertModule(
            "i.asf.download",
            flags="w",
            platform="Sentinel-1",
            beam_mode="IW",
            processinglevel="GRD_HD",
            start="2023-03-15T00:00:00Z",
            end="2023-03-16T00:00:00Z",
            aoi="./data/roi.geojson",
            output_directory=self.tempdir,
            check_scenes="all",
            nprocs=2,
            verbose=True,
        )

    def test_asf_download_flag_l(self):
        """Test download and import of Sentinel-1 with flag l - list only"""
        self.assertModule(
            "i.asf.download",
            flags="l",
            platform="Sentinel-1",
            beam_mode="IW",
            processinglevel="GRD_HD",
            start="2023-03-15T00:00:00Z",
            end="2023-03-16T00:00:00Z",
            aoi="./data/roi.geojson",
            verbose=True,
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
