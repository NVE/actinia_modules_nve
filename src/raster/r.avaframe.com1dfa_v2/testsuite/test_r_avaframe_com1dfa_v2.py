"""Test r.avaframe.comdfa_v2


(C) 2022 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

import os
from subprocess import PIPE

import grass.script as gs
from grass.gunittest.case import TestCase


class TestAvaframeV2(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.use_temp_region()

        # Import testdata
        cls.runModule(
            "r.import",
            input="data/DTM_10m.tif",
            output="DTM_10m",
            resolution=10,
            overwrite=True,
        )

        cls.runModule("g.region", raster="DTM_10m", align="DTM_10m")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        cls.del_temp_region()

    def tearDown(self):
        """Remove generated data"""

    def test_avaframe_v2(self):
        """Test avaframe v2 with entrainment, resistance and
        multiple release thicknesses"""
        avaframe_run = gs.start_command(
            "r.avaframe.com1dfa_v2",
            flags="l",
            id="2",
            url="https://gis3.nve.no/arcgis/rest/services/featureservice/AlarmInput/FeatureServer",
            release_area_layer_id="0",
            entrainment_area_layer_id="1",
            resistance_area_layer_id="2",
            elevation="DTM_10m",
            buffer=3000,
            nprocs=2,
            # ppr="ppr",
            # pft="pft",
            # pfv="pfv",
            export_directory="./",
            stderr=PIPE,
            stdout=PIPE,
        )
        stdout, stderr = avaframe_run.communicate()
        stderr = stderr.decode("utf8").lower()
        gs.warning(str(stdout))
        gs.warning(str(stderr))
        self.assertFalse("error" in stderr or "traceback" in stderr)


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
