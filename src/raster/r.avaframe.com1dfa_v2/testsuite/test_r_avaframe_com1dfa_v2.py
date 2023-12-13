"""Test t.rast.reclass

(C) 2022 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""
import os

import grass.pygrass.modules as pymod
from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.use_temp_region()

        # Import testdata
        cls.runModule(
            "r.in.gdal",
            input="data/DTM_10m.tif",
            output="DTM_10m",
            overwrite=True,
        )

        cls.runModule("g.region", raster="DTM_10m", align="DTM_10m")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        cls.del_temp_region()

    def tearDown(self):
        """Remove generated data"""
        pass

    def test_reclass_with_null_maps(self):
        """Reclassify and register also maps with only NoData"""
        self.assertModule(
            "r.avaframe.com1dfa_v2",
            flags="er",
            id="1",
            url="https://gis3.nve.no/arcgis/rest/services/featureservice/AlarmInput/FeatureServer",
            release_area_layer_id="0",
            entrainment_area_layer_id="1",
            resistance_area_layer_id="2",
            elevation="DTM_10m",
            buffer=3000,
            nprocs=2,
            ppr="ppr",
            pft="pft",
            pfv="pfv",
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
