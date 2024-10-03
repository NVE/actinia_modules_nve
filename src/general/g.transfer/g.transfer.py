#! /usr/bin/python3
"""
MODULE:    g.transfer
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Move or copy files or directories from source to target
COPYRIGHT: (C) 2024 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General
Public License (>=v2). Read the file COPYING that
comes with GRASS for details.
"""

# %module
# % description: Move or copy files or directories from source to target
# % keyword: general
# % keyword: file
# % keyword: directory
# % keyword: move
# % keyword: copy
# %end

# %option
# % key: source
# % description: Path to source of files or directories to transfer (supports wildcards (*))
# % required: yes
# %end

# %option G_OPT_M_DIR
# % key: target
# % description: Path to target to transfer files or directories to
# % required: yes
# %end

# %option G_OPT_M_NPROCS
# %end

# %flag
# % key: m
# % label: Move files or directories (default is copy)
# % description: Move files or directories (default is copy)
# %end

# %flag
# % key: o
# % label: Overwrite existing files
# % description: Overwrite existing files
# %end

# ruff: noqa: PTH207

import shutil
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import grass.script as gs


def transfer(source, target=None, move=False, overwrite=False):
    """Function to transfer files from source to target"""
    source = Path(source)
    if source.is_file():
        target_name = target / source.name
        if target_name.exists():
            if not overwrite:
                gs.fatal(_("Target <{}> exists. Please use the overwrite flag."))
            target_name.unlink()
        if move:
            gs.verbose(
                _("Moving <{source}> to <{target}>").format(
                    source=source, target=target
                )
            )
            shutil.move(source, target)
            return
        shutil.copy2(source, target)
        return
    if source.is_dir():
        gs.verbose(
            _("{method} directory tree <{source}> to <{target}>").format(
                method="Moving" if move else "Copying", source=source, target=target
            )
        )
        target_dir = target / source.name
        if target_dir.exists() and not overwrite:
            gs.fatal(
                _("Target <{}> exists. Please use the overwrite flag.").format(
                    str(target_dir)
                )
            )
        shutil.copytree(source, target_dir, dirs_exist_ok=True)
        if move:
            shutil.rmtree(source)
        return
    gs.verbose(
        _("Copying file <{source}> to <{target}>").format(source=source, target=target)
    )
    shutil.copy2(source, target)


def main():
    """Do the main work"""
    paths_to_transfer = glob(options["source"])
    if not paths_to_transfer:
        gs.warning(
            _("Nothing found to transfer with source <{}>.").format(options["source"])
        )
        sys.exit(0)

    target_directory = Path(options["target"]) if options["target"] else Path.cwd()

    if target_directory.exists() and target_directory.is_file():
        gs.fatal(
            _("Target <{}> exists and is not a directory.").format(options["target"])
        )
    target_directory.mkdir(exist_ok=True, parents=True)

    transfer_function = partial(
        transfer, target=target_directory, move=flags["m"], overwrite=flags["o"]
    )
    nprocs = int(options["nprocs"])

    if nprocs > 1:
        with Pool(nprocs) as pool:
            pool.map(transfer_function, paths_to_transfer)
    else:
        for transfer_path in paths_to_transfer:
            transfer_function(transfer_path)


if __name__ == "__main__":
    options, flags = gs.parser()
    sys.exit(main())
