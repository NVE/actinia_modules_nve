"""Test t.rast.stats

(C) 2023 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""


from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAreaStats(TestCase):
    """Test case for t.rast.stats"""

    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
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
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="n",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|100
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|200
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|300
""",
        )

    def test_basic_stats_percent(self):
        """Test basic area statistics in percent"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="p",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|100|101.05%
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|200|101.05%
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|300|101.05%
""",
        )

    def test_basic_stats_m2(self):
        """Test basic area statistics in m2"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="a",
            input="A",
            nprocs=1,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|100|9600.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|200|9600.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|300|9600.000000
""",
        )

    def test_stats_with_zone(self):
        """Test area statistics with one zone map"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="n",
            input="A",
            zone="zone_x",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0|100
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1|100
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2|100
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3|100
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0|200
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1|200
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2|200
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3|200
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0|300
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1|300
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2|300
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3|300""",
        )

    def test_stats_with_zone_and_label(self):
        """Test area statistics with one zone map and label"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="nla",
            input="A",
            zone="zone_x",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0||100||2400.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1||100||3200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2||100||3200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3||100||800.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0||200||2400.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1||200||3200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2||200||3200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3||200||800.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0||300||2400.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1||300||3200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2||300||3200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3||300||800.000000
""",
        )

    def test_stats_with_zone_and_label(self):
        """Test area statistics with one zone map, label and header"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="hnla",
            input="A",
            zone="zone_x",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """map|start|end|zone|zone_label|raster_value|raster_value_label|area_m2
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0||100||2400.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1||100||3200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2||100||3200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3||100||800.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0||200||2400.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1||200||3200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2||200||3200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3||200||800.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0||300||2400.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1||300||3200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2||300||3200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3||300||800.000000
""",
        )

    def test_stats_with_two_zones(self):
        """Test area statistics with two zone maps"""
        stats_module = SimpleModule(
            "t.rast.stats",
            verbose=True,
            flags="na",
            input="A",
            zone="zone_x,zone_y",
            nprocs=2,
        )
        self.assertModule(stats_module.run())
        self.assertLooksLike(
            stats_module.outputs.stdout,
            """a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0|0|100|900.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0|1|100|1200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|0|2|100|300.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1|0|100|1200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1|1|100|1600.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|1|2|100|400.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2|0|100|1200.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2|1|100|1600.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|2|2|100|400.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3|0|100|300.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3|1|100|400.000000
a1@...|2001-01-15 12:05:45|2001-01-16 12:05:45|3|2|100|100.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0|0|200|900.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0|1|200|1200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|0|2|200|300.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1|0|200|1200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1|1|200|1600.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|1|2|200|400.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2|0|200|1200.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2|1|200|1600.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|2|2|200|400.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3|0|200|300.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3|1|200|400.000000
a2@...|2001-01-16 12:05:45|2001-01-17 12:05:45|3|2|200|100.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0|0|300|900.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0|1|300|1200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|0|2|300|300.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1|0|300|1200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1|1|300|1600.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|1|2|300|400.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2|0|300|1200.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2|1|300|1600.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|2|2|300|400.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3|0|300|300.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3|1|300|400.000000
a3@...|2001-01-17 12:05:45|2001-01-18 12:05:45|3|2|300|100.000000
""",
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
