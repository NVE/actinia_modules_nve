"""Test i.asf.download for Sentinel-1

(C) 2023 by NVE, Yngve Antonsen and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: 
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
        aoi = shapely.wkt.dumps(gpd.read_file('./roi.geojson').iloc[0].geometry)
        test_product = "ASF_S1_download"
        self.assertModule(
            "i.asf.download",
            product_type="Sentinel-1",
            beamMode="IW",
            processingLevel="GRD_HD",
            start="2023-03-15T00:00:00Z",
            end="2023-03-16T00:00:00Z",
            intersectsWith=aoi,
            nprocs=2,
            memory=2048,
            token="./token.txt", #Må skrive om til å bruke github-secrets
            output_directory=self.tempdir,
            output=test_product,

        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_asf_download_flag_c(self):
        """Test download and import of Sentinel-1 with flag c - checksum-test"""
        aoi = shapely.wkt.dumps(gpd.read_file('./roi.geojson').iloc[0].geometry)
        test_product = "ASF_S1_download_flag_c"
        self.assertModule(
            "i.asf.download",
            flags="c",
            product_type="Sentinel-1",
            beamMode="IW",
            processingLevel="GRD_HD",
            start="2023-03-15T00:00:00Z",
            end="2023-03-16T00:00:00Z",
            intersectsWith=aoi,
            nprocs=2,
            memory=2048,
            token="./token.txt", #Må skrive om til å bruke github-secrets
            output_directory=self.tempdir,
            output=test_product,

        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)
        
    def test_asf_download_flag_c(self):
            """Test download and import of Sentinel-1 with flag l - list only"""
            aoi = shapely.wkt.dumps(gpd.read_file('./roi.geojson').iloc[0].geometry)
            test_product = "ASF_S1_download_flag_l"
            self.assertModule(
                "i.asf.download",
                flags="l",
                product_type="Sentinel-1",
                beamMode="IW",
                processingLevel="GRD_HD",
                start="2023-03-15T00:00:00Z",
                end="2023-03-16T00:00:00Z",
                intersectsWith=aoi,
                nprocs=2,
                memory=2048,
                token="./token.txt", #Må skrive om til å bruke github-secrets
                output_directory=self.tempdir,
                output=test_product,

            )
            info = SimpleModule(
                "t.info",
                flags="g",
                input=test_product,
            ).run()
            print(info.outputs.stdout)


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
