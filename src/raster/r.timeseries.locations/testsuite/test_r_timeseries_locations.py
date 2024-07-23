"""Test t.rast.reclass

(C) 2022 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

import os

from grass.gunittest.case import TestCase


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.use_temp_region()
        cls.runModule("g.region", s=0, n=80, w=0, e=120, b=0, t=50, res=10, res3=10)
        cls.runModule("r.mapcalc", expression="a1 = 100", overwrite=True)

        # Import testdata
        cls.runModule(
            "r.in.gdal",
            input="data/DTM_10m.tif",
            output="DTM_10m",
            overwrite=True,
        )

        cls.runModule(
            "v.in.ogr",
            input="./data/releasearea.gpkg",
            output="releasearea",
            overwrite=True,
        )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        cls.del_temp_region()
        # cls.runModule("t.remove", flags="df", type="strds", inputs="A")

    def tearDown(self):
        """Remove generated data"""
        # self.runModule("t.remove", flags="df", type="strds", inputs="B")

    def test_reclass_with_null_maps(self):
        """Reclassify and register also maps with only NoData"""
        self.assertModule(
            "r.timeseries.locations",
            input="",
            elevation="",
            overwrite=True,
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
