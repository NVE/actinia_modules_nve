#! /usr/bin/python3
"""
MODULE:    g.unzip
AUTHOR(S): Stefan Blumentrath
PURPOSE:	 Unzip zip-files in a directory in parallel
COPYRIGHT: (C) 2023-2024 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General
Public License (>=v2). Read the file COPYING that
comes with GRASS for details.
"""

# %module
# % description: Unzip zip-files in a directory in parallel
# % keyword: general
# % keyword: zip
# % keyword: unzip
# % keyword: unpack
# %end

# %option
# % key: input
# % type: string
# % description: Path to the input zip-file or directory with zip-files to extract
# % required: yes
# %end

# %option G_OPT_M_DIR
# % key: output
# % description: Path to the output directory to extract zip-files to
# % required: no
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: s
# % label: Skip corrupt zip-file(s)
# % description: Skip corrupt zip-file(s)
# %end

# %flag
# % key: r
# % label: Remove zip-file(s) after extraction
# % description: Remove zip-file(s) after extraction
# %end

import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile

import grass.script as gs


def unzip_file(file_path, out_dir=None, remove=False):
    """Unzip file to output directory if given or current working directory
    and remove unpacked zip-file if requested
    Unzipping preserves modification time"""

    gs.verbose(_("Unzipping {}").format(str(file_path)))

    out_dir = Path("./") if not out_dir else Path(out_dir)

    try:
        with ZipFile(file_path) as zip_file_object:
            for zipped_file in zip_file_object.infolist():
                file_name, file_date_time = (
                    zipped_file.filename.lstrip("/").lstrip("\\"),
                    zipped_file.date_time,
                )
                out_file_name = out_dir / file_name
                # Create directory if path in zipfile is a directory
                if zipped_file.is_dir():
                    out_file_name.mkdir(parents=True, exist_ok=True)
                else:
                    # Create parent directory if needed
                    if not out_file_name.parent.exists():
                        out_file_name.parent.mkdir(parents=True, exist_ok=True)
                    # Extract file
                    with zip_file_object.open(zipped_file) as zip_content:
                        out_file_name.write_bytes(zip_content.read())
                file_date_time = time.mktime((*file_date_time, 0, 0, -1))
                os.utime(out_file_name, (file_date_time, file_date_time))
    except OSError:
        if flags["s"]:
            return 0
        else:
            gs.fatal(_("Could not unzip file <{}>.").format(file_path))

    if not remove:
        return 0

    file_path.unlink()
    return 0


def main():
    """Do the main work"""
    input_path = Path(options["input"])

    if not input_path.exists():
        gs.fatal(_("Input file or directory <{}> not found").format(str(input_path)))
    elif input_path.is_dir():
        output_directory = input_path
        input_files = list(input_path.glob("*.[zZ][iI][pP]"))
    elif input_path.is_file():
        output_directory = input_path.parent
        input_files = [input_path]

    if len(input_files) <= 0:
        gs.warning(
            _("No zip-files found in input directory <{}>").format(str(input_path))
        )
        sys.exit(0)

    if options["output"]:
        output_directory = Path(options["output"])

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        gs.fatal(_("Cannot create output directory <{}>").format(str(output_directory)))
    if not os.access(output_directory, os.W_OK):
        gs.fatal(
            _("Output directory <{}> is not writeable").format(str(output_directory))
        )

    unzip = partial(unzip_file, out_dir=str(output_directory), remove=flags["r"])

    nprocs = min(int(options["nprocs"]), len(input_files))
    if nprocs == 1:
        for zip_file in input_files:
            unzip(zip_file)
    else:
        gs.verbose(
            _("Unzipping {} files to {} using {} parallel processes").format(
                len(input_files), str(output_directory), nprocs
            )
        )
        with Pool(nprocs) as pool:
            pool.map(unzip, input_files)


if __name__ == "__main__":
    options, flags = gs.parser()
    sys.exit(main())
