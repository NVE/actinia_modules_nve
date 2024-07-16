"""Test i.satskred

(C) 2023 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

import os

import grass.script as gs

from grass.gunittest.case import TestCase


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        cls.tempdir = gs.tempdir()
        cls.use_temp_region()
        gs.run_command(
            "g.region",
            w=659955,
            n=7719679,
            e=671015,
            s=7701760,
            res=20,
            flags="ap",
        )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        cls.del_temp_region()
        gs.utils.try_rmdir(cls.tempdir)

    def test_satskred(self):
        """Test satskred run
        This currently fails due to missing test data"""
        self.assertModuleFail(
            "i.satskred",
            input="./data/Sentinel_1",
            elevation="./data/dtm20m.tif",
            mask_directory="./data/runoutmasks/",
            start="2019-11-24",
            end="2020-06-06",
            output_directory=self.tempdir,
            verbose=True,
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
