"""Test i.pytorch.predict

(C) 2023 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

import unittest

import grass.script as gs
from grass.gunittest.case import TestCase


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the working environment
        and import data for test case"""
        cls.tempdir = gs.tempdir()
        # Import data
        # gs.run_command(
        #     "r.in.gdal",
        #     flags="o",
        #     input="data/S3_SLSTR_reflectance.tif",
        #     output="S3_SLSTR_reflectance",
        #     overwrite=True,
        # )
        # # Add semantic labels
        # for band in range(5):
        #     band = band + 1
        #     # Skip band 4
        #     gs.run_command(
        #         "r.support",
        #         map=f"S3_SLSTR_reflectance.{band}",
        #         semantic_label=f"S{band if band < 4 else band + 1}_reflectance_an",
        #         overwrite=True,
        #     )
        # # Create imagery group
        # gs.run_command(
        #     "i.group",
        #     group="S3_SLSTR_test_case",
        #     input=[f"S3_SLSTR_reflectance.{band + 1}" for band in range(5)],
        #     overwrite=True,
        # )

        # cls.use_temp_region()
        # gs.run_command(
        #     "g.region",
        #     raster="S3_SLSTR_reflectance.1",
        #     n=7719679,
        #     e=671015,
        #     s=7701760,
        #     res=20,
        #     flags="ap",
        # )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        cls.del_temp_region()
        gs.run_command("g.remove", type="raster", pattern="S3_SLSTR_reflectance*")

    @unittest.skip("Skipping due to lack of test-data")
    def test_torch_prediction_no_tiles(self):
        """Test i.pytorch.predict runs as expected
        with just one tile (=without tiling)
        This currently fails due to missing test data"""
        self.assertModuleFail(
            "i.pytorch.predict",
            input="S3_SLSTR_test_case",
            tile_size="512,512",
            overlap=128,
            output="S3_SLSTR_reflectance_untiled",
            configuration="data/fsc.json",
            model="data/fsc.pt",
            nprocs=6,
            model_code="data/",
            verbose=True,
        )

    @unittest.skip("Skipping due to lack of test-data")
    def test_torch_prediction_with_tiles(self):
        """Test i.pytorch.predict runs as expected
        with just one tile (=without tiling)
        This currently fails due to missing test data"""
        self.assertModuleFail(
            "i.pytorch.predict",
            input="S3_SLSTR_test_case",
            tile_size="405,512",
            overlap=128,
            output="S3_SLSTR_reflectance_untiled",
            configuration="data/fsc.json",
            model="data/fsc.pt",
            nprocs=6,
            model_code="data/",
            verbose=True,
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
