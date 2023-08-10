"""Test i.sentinel1.pyrosargeocode for Sentinel-1

(C) 2023 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""
import os

import grass.script as gs

from grass.gunittest.case import TestCase


class TestPyrosarGeocoding(TestCase):
    """Basic class covering all test cases"""

    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.tempdir = gs.tempdir()
        cls.output_dir = gs.tempdir()

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        gs.utils.try_rmdir(cls.tempdir)

    def test_asf_download(self):
        """Test geocoding of Sentinel-1 data with pyrosar
        Tests cannot succeed due to missing test data
        asserting fail until that is solved
        """
        self.assertModuleFail(
            "i.sentinel1.pyrosargeocode",
            input="S1A_XXXXXXXXXXX",
            flags="fndm",
            speckle_filter="refined_lee",
            temporary_directory=self.tempdir,
            nprocs=2,
            elevation="DTM_10m@DTM",
            aoi="./data/roi.geojson",
            output_directory=self.output_dir,
            verbose=True,
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
