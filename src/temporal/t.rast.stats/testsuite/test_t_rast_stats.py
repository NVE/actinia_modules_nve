"""Test t.rast.reclass

(C) 2022 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""
import os

import grass.pygrass.modules as pymod
import grass.temporal as tgis
from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")
        tgis.init()
        cls.use_temp_region()
        cls.runModule("g.region", s=0, n=80, w=0, e=120, b=0, t=50, res=10, res3=10)
        cls.runModule("r.mapcalc", expression="a1 = 100", overwrite=True)
        cls.runModule("r.mapcalc", expression="a2 = 200", overwrite=True)
        cls.runModule("r.mapcalc", expression="a3 = 300", overwrite=True)
        cls.runModule(
            "r.mapcalc", expression="zone_x = int(col() / 4.0)", overwrite=True
        )
        cls.runModule(
            "r.mapcalc", expression="zone_y = int(row() / 4.0)", overwrite=True
        )

        cls.runModule(
            "t.create",
            type="strds",
            temporaltype="absolute",
            output="A",
            title="A test",
            description="A test",
            overwrite=True,
        )

        cls.runModule(
            "t.register",
            flags="i",
            type="raster",
            input="A",
            maps="a1,a2,a3",
            start="2001-01-15 12:05:45",
            increment="1 days",
            overwrite=True,
        )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        cls.del_temp_region()
        cls.runModule("t.remove", flags="df", type="strds", inputs="A")

    def test_basic_stats(self):
        """Test basic area statistics"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="n",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")

    def test_basic_stats_percent(self):
        """Test basic area statistics in procent"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="p",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")

    def test_basic_stats_m2(self):
        """Test basic area statistics in m2"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="p",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")

    def test_stats_with_zone(self):
        """Test area statistics with one zone map"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="n",
            input="A",
            zone="zone_x",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")

    def test_stats_with_zone_and_label(self):
        """Test area statistics with one zone map"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="nl",
            input="A",
            zone="zone_x",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")

    def test_stats_with_two_zones(self):
        """Test area statistics with one zone map"""
        stats_module = self.SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="n",
            input="A",
            zone="zone_x,zone_y",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(info.outputs.stdout, "")


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
