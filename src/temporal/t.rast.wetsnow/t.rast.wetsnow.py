#!/usr/bin/env python3

############################################################################
#
# MODULE:       t.rast.wetsnow
# AUTHOR(S):    Stefan Blumentrath
#
# PURPOSE:      Detect wet snow from Sentinel-1 GRDH IW time series and reference data
# SPDX-FileCopyrightText: 2026 GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
#
#############################################################################

# %module
# % description: Detect wet snow from Sentinel-1 GRDH IW time series and reference data.
# % keyword: temporal
# % keyword: raster
# % keyword: wetsnow
# % keyword: Sentinel-1
# %end

# %option G_OPT_STRDS_INPUT
# %end

# %option G_OPT_STRDS_OUTPUT
# %end

# %option
# % key: basename
# % type: string
# % label: Basename of the new generated output maps
# % description: Either a numerical suffix or the start time (s-flag) separated by an underscore will be attached to create a unique identifier
# % required: yes
# % multiple: no
# %end

# %option
# % key: reference_pattern
# % type: string
# % label: Pattern to find groups with reference data
# % description: Pattern to find groups with reference data, may include mapset (e.g. "Sentinel_1_reference_*@Sentinel_1_reference")
# % required: yes
# % multiple: no
# %end

# %option
# % key: size
# % type: integer
# % label: Neighborhood size for noise filtering
# % description: Neighborhood size for noise filtering (must be odd, default: 3)
# % guisection: Settings
# % required: no
# % multiple: no
# % answer: 5
# %end

# %option
# % key: detection_thresholds
# % type: double
# % label: Lower and upper threshold for wet snow detection in DBi backscatter difference
# % description: Lower and upper threshold for wet snow detection in DBi backscatter difference (default: -12.0,-1.4)
# % guisection: Settings
# % required: no
# % multiple: yes
# % answer: -12.0,-1.4
# %end

# %option
# % key: nodata_threshold
# % type: double
# % label: Nodata threshold
# % description: Lowest value for valid backscatter data (default: -30.0)
# % guisection: Settings
# % required: no
# % multiple: no
# % answer: -30.0
# %end

# %option
# % key: k
# % type: double
# % label: Weight factor for local incidence angle
# % description: Weight factor for local incidence angle, only used if reference data do not contain weight raster (default: 0.5)
# % guisection: Settings
# % required: no
# % multiple: no
# % answer: 0.5
# %end

# %option
# % key: mode_size
# % type: integer
# % label: Neighborhood size for mode filtering of the final result
# % description: Neighborhood size for mode filtering of the final result (must be odd, default: 3)
# % guisection: Settings
# % required: no
# % multiple: no
# % answer: 3
# %end

# %option
# % key: method
# % type: string
# % label: Method to be used for noise filtering
# % description: Method to be used for noise filtering (default: median)
# % guisection: Settings
# % required: yes
# % multiple: no
# % options: average,median,quart1,quart3,perc90
# % answer: median
# %end

# %option G_OPT_MEMORYMB
# %answer: 2048
# %end

# %option G_OPT_M_NPROCS
# %end

# %option G_OPT_T_WHERE
# %end

# %flag
# % key: e
# % description: Extend to existing output STRDS (requires overwrite flag)
# %end

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from sqlite3 import Row

    from grass.temporal import SpaceTimeRasterDataset

import atexit
import os
import re
from functools import partial
from multiprocessing import Pool
from operator import call
from tempfile import NamedTemporaryFile

import grass.script as gs
from grass.tools import Tools

TEMP_NAME = gs.tempname(12)


def cleanup() -> None:
    """Remove temporary raster maps.

    Removes all raster maps matching the global ``TEMP_NAME`` prefix,
    including per-task temporary maps derived from it.
    """
    tools = Tools()
    tools.g_remove(type="raster", pattern=f"{TEMP_NAME}*", flags="f", quiet=True)


def _reduce_noise(
    raster_map: str,
    filter_size: int = 5,
    resolution: int = 100,
    *,
    nodata_threshold: float | None = None,
    method: str = "median",
    nprocs: int = 1,
    memory: int = 2048,
    temp_name: str = TEMP_NAME,
) -> str:
    """Reduce noise in a raster map by reclassifying, resampling and filtering.

    Optionally masks out values below a nodata threshold, resamples to a
    coarser resolution using ``method`` if the map is finer than
    ``resolution``, and applies a neighborhood filter of ``filter_size``.

    :param str raster_map: Name of the input raster map
    :param int filter_size: Neighborhood size for noise filtering (must be
        odd), filtering is skipped if 0
    :param int resolution: Target resolution in map units
    :param float nodata_threshold: Lowest value considered valid; lower
        values are set to null, no reclassification if None
    :param str method: Aggregation/filtering method (e.g. median, average)
    :param int nprocs: Number of parallel processes to use for tool calls
    :param int memory: Memory in MB to use for tool calls
    :param str temp_name: Prefix used to name temporary raster maps
    :return: Name of the resulting (possibly unmodified) raster map
    :rtype: str
    """
    tools = Tools()
    result_name = raster_map
    raster_map_basename = next(iter(raster_map.split("@")))
    rmap_info = tools.r_info(map=raster_map, format="json").json
    aggregate_resolution = (
        rmap_info["ewres"] < resolution or rmap_info["nsres"] < resolution
    )
    input_map = raster_map_basename
    if (
        (nodata_threshold and nodata_threshold > rmap_info["min"])
        or filter_size > 0
        or aggregate_resolution
    ):
        reclassed_map = f"{temp_name}_{raster_map_basename}_rc"
        with gs.RegionManager(raster=raster_map, align=raster_map):
            tools.r_mapcalc(
                expression=f"{reclassed_map}=if({input_map} >= {nodata_threshold}, {input_map}, null())",
                nprocs=nprocs,
                overwrite=True,
            )
        result_name = reclassed_map
        input_map = reclassed_map
    if aggregate_resolution:
        resampled_map = (
            f"{raster_map_basename}_{resolution}m"
            if filter_size == 0
            else f"{temp_name}_{raster_map_basename}"
        )
        tools.r_resamp_stats(
            flags="w",
            input=input_map,
            output=resampled_map,
            method=method,
            nprocs=nprocs,
            memory=memory,
            overwrite=True,
        )
        result_name = resampled_map
        input_map = resampled_map
    if filter_size > 0:
        result_name = (
            f"{temp_name}_{raster_map_basename}_{resolution}m_{method}{filter_size:02}"
        )
        tools.r_neighbors(
            input=input_map,
            output=result_name,
            method=method,
            size=filter_size,
            nprocs=nprocs,
            memory=memory,
            overwrite=True,
        )
    return result_name


def compute_wet_snow(
    track: int,
    timestamps: tuple[datetime, datetime | None],
    vv_map: str,
    vh_map: str,
    reference_maps: dict,
    *,
    method: str = "median",
    median_filter: int = 5,
    mode_filter: int = 3,
    k: float = 0.5,
    upper_detection_threshold: float = -1.4,
    lower_detection_threshold: float = -12.0,
    nodata_threshold: float = -30.0,
    basename: str = "Sentinel_1_WetSnow",
    nprocs: int = 1,
    memory: int = 2048,
    overwrite: bool = True,
    temp_name: str = TEMP_NAME,
    mapset: str | None = None,
) -> str:
    """Compute a wet snow map for one track and time step, under a raster mask.

    Applies noise reduction to the VV and VH input maps, combines their
    difference to the reference maps weighted by local incidence angle,
    and classifies the result using the detection thresholds. Intermediate
    maps named with ``temp_name`` are removed before returning.

    :param int track: Track number the input maps belong to
    :param tuple[datetime, datetime | None] timestamps: Start and end time
        of the time step (end may be None)
    :param str vv_map: Name of the input VV backscatter raster map
    :param str vh_map: Name of the input VH backscatter raster map
    :param dict reference_maps: Reference maps for the track, with keys
        ``VV``, ``VH``, ``linc``, ``linc_weight`` and
        ``mask_high_resolution``
    :param str method: Method used for noise filtering
    :param int median_filter: Neighborhood size for noise filtering
    :param int mode_filter: Neighborhood size for mode filtering of the
        final result, skipped if 0
    :param float k: Weight factor for local incidence angle, used only if
        reference data do not contain a weight raster
    :param float upper_detection_threshold: Upper threshold for wet snow
        detection in DBi backscatter difference
    :param float lower_detection_threshold: Lower threshold for wet snow
        detection in DBi backscatter difference
    :param float nodata_threshold: Lowest value for valid backscatter data
    :param str basename: Basename used for the resulting output map
    :param int nprocs: Number of parallel processes to use for tool calls
    :param int memory: Memory in MB to use for tool calls
    :param bool overwrite: Whether to overwrite existing maps
    :param str temp_name: Prefix used to name and later remove temporary
        raster maps created for this track and time step
    :param str mapset: Mapset to qualify the returned map name with,
        current mapset is used if None
    :return: TGIS registration line(s) for the resulting map(s), in the
        format ``name@mapset|start_time[|end_time]|semantic_label``
    :rtype: str
    """
    gs.verbose(_("Computing WetSnow for track %i at %s") % (track, f"{timestamps[0]}"))
    env = os.environ.copy()
    tools = Tools(env=env)
    if mapset is None:
        mapset = gs.gisenv()["MAPSET"]
    # Noise reduction
    # Get reference_maps
    vv_map_reference = reference_maps["VV"]
    vh_map_reference = reference_maps["VH"]
    linc_weight = reference_maps["linc_weight"]
    linc = reference_maps["linc"]
    current_region = tools.g_region(flags="up", format="json").json
    resolution = int(current_region["nsres"])
    # Apply mask to all following operations
    with gs.MaskManager(mask_name=reference_maps["mask_high_resolution"], env=env):
        vv_map = _reduce_noise(
            vv_map,
            median_filter,
            resolution,
            nodata_threshold=nodata_threshold,
            method=method,
            nprocs=nprocs,
            memory=memory,
            temp_name=temp_name,
        )
        vh_map = _reduce_noise(
            vh_map,
            median_filter,
            resolution,
            nodata_threshold=nodata_threshold,
            method=method,
            nprocs=nprocs,
            memory=memory,
            temp_name=temp_name,
        )
        start_time, end_time = timestamps
        result_map = (
            f"{basename if mode_filter > 0 else temp_name}_"
            f"{resolution}m_{track}_{start_time.strftime('%Y%m%d%H%M%S')}"
        )
        difference_vv = f"{vv_map} - {vv_map_reference}"
        difference_vh = f"{vh_map} - {vh_map_reference}"
        mc_expression = ""
        if linc_weight:
            difference_combined = f"({linc_weight} * ({difference_vh})) + (1 - {linc_weight}) * ({difference_vv})"
        else:
            linc_weighting = (
                f"W=if({linc} >= 45.0, {k}, if({linc} <= 20.0, 1.0, "
                f"float(1 + (45.0 - {linc}) / float(45.0 - 20.0))*{k}))"
            )
            difference_combined = f"W * {difference_vh} + (1 - W) * {difference_vv}"
            mc_expression = f"eval({linc_weighting})\n"
        mc_expression += (
            f"{result_map}=if(({difference_combined}) < {upper_detection_threshold} && "
            f"({difference_combined}) > {lower_detection_threshold}, 1, 0)"
        )
        tools.r_mapcalc(
            expression=mc_expression,
            nprocs=nprocs,
            overwrite=overwrite,
        )
        if mode_filter > 0:
            tools.r_neighbors(
                input=result_map,
                output=f"{result_map}_mode_{mode_filter:02}",
                method="mode",
                size=mode_filter,
                nprocs=nprocs,
                memory=memory,
                overwrite=overwrite,
            )
    # Remove temporary data
    tools.g_remove(type="raster", pattern=f"{temp_name}*", flags="f", quiet=True)
    if end_time:
        return f"{result_map}@{mapset}|{start_time}|{end_time}|S1_WetSnow_{track}\n"
    return f"{result_map}@{mapset}|{start_time}|S1_WetSnow_{track}\n"


def extract_track_number(name: str, pattern: str) -> int | None:
    """Extract track number from group name using pattern.

    Using unnamed groups. Alternative would be users specifying a full re pattern.

    :param str name: Name of the imagery group to match
    :param str pattern: Pattern with a single ``*`` wildcard marking the
        part of the name that contains the track number
    :return: Extracted track number, or None if no match or no digits found
    :rtype: int | None
    """
    regex = re.escape(pattern).replace(r"\*", r"(.+)")
    match = re.fullmatch(regex, name)
    if not match:
        return None
    wildcard = match.group(1)
    num = re.search(r"\d{1,3}", wildcard)
    return int(num.group()) if num else None


def get_reference_data(pattern: str) -> dict:
    """Get image groups with reference data as dict.

    :param str pattern: Pattern to find groups with reference data, may
        include mapset (e.g. ``Sentinel_1_reference_*@Sentinel_1_reference``)
    :return: Mapping of track number to the imagery group content, as
        returned by :func:`grass.script.imagery.group_to_dict`
    :rtype: dict
    """
    groups_dict = {}
    tools = Tools()
    mapset = None
    pattern_no_mapset = pattern
    if "@" in pattern:
        pattern_no_mapset, mapset = pattern.split("@", 1)
    groups = tools.g_list(
        type="group",
        pattern=pattern_no_mapset,
        mapset=mapset,
        format="json",
    ).json
    for g in groups:
        group = g["fullname"]
        track = extract_track_number(group, pattern)
        if track is None:
            gs.warning(_("Group %s does not contain a track number.") % group)
            continue
        groups_dict.update({track: gs.imagery.group_to_dict(group)})
    if not groups_dict:
        gs.fatal(_("No reference data found with pattern <%s>") % pattern)
    return groups_dict


def parse_semantic_label(semantic_label: str) -> tuple[str, int] | None:
    """Extract track number and polarization from semantic label.

    :param str semantic_label: Semantic label of a registered map, expected
        to contain a polarization (VV/VH) and a track number
    :return: Tuple of polarization and track number, or None if either
        could not be found in the semantic label
    :rtype: tuple[str, int] | None
    """
    name = semantic_label.replace("S1", "")
    pol = re.search(r"(?P<polarization>VV|VH|vv|vh)", name)
    track_str = re.search(r"(?P<track>\d{1,3})", name)
    if not pol:
        return None
    if not track_str:
        return None
    return pol.group("polarization"), int(track_str.group("track"))


def group_input_maps(map_list: list[Row]) -> dict:
    """Group registered maps by track and time step, keyed by polarization.

    :param list[sqlite3.Row]: Registered maps as returned by
        :func:`SpaceTimeRasterDataset.get_registered_maps`
    :return: Mapping of track number to a mapping of
        ``(start_time, end_time)`` to a mapping of polarization to map id
    :rtype: dict
    """
    scenes_dict = {}
    for m in map_list:
        # SQLite3 Row object do not support in-syntax
        if "semantic_label" not in m.keys():  # noqa: SIM118
            gs.warning(_("No semantic label returned for map %s") % m["id"])
            continue
        pol_and_track = parse_semantic_label(m["semantic_label"])
        if not pol_and_track:
            gs.warning(
                _(
                    "Could not extract polarization and track from semantic"
                    " label %s of map %s",
                )
                % (m["semantic_label"], m["id"]),
            )
            continue
        polarization, track = pol_and_track
        if track in scenes_dict:
            if (m["start_time"], m["end_time"]) in scenes_dict[track]:
                scenes_dict[track][m["start_time"], m["end_time"]][polarization] = m[
                    "id"
                ]
            else:
                scenes_dict[track][m["start_time"], m["end_time"]] = {
                    polarization: m["id"],
                }
        else:
            scenes_dict[track] = {
                (m["start_time"], m["end_time"]): {polarization: m["id"]},
            }
    return scenes_dict


def check_complete_polarizations(temporal_extents: dict) -> dict:
    """Keep only time steps that have both VV and VH polarization.

    :param dict temporal_extents: Mapping of ``(start_time, end_time)`` to
        a mapping of polarization to map id, as produced by
        :func:`group_input_maps`
    :return: Same mapping, with time steps missing VV or VH removed
    :rtype: dict
    """
    complete_extents = {}
    for temporal_extent, polarizations in temporal_extents.items():
        missing_polarization = {"VV", "VH"} - set(polarizations)
        if missing_polarization:
            gs.warning(
                _("Missing %s polarization for time step %s. Skipping...")
                % (missing_polarization, f"{temporal_extent[0]}"),
            )
            continue
        complete_extents[temporal_extent] = polarizations
    return complete_extents


def register_in_tgis(
    map_list: list[str],
    output: str,
    input_stds: SpaceTimeRasterDataset,
    *,
    extend: bool = True,
    overwrite: bool = True,
) -> None:
    """Register results in TGIS DB.

    Creates the output STRDS if it does not yet exist, or recreates it if
    it exists and ``extend`` is False, then registers the given maps in it.

    :param list[str]: TGIS registration lines, one per map, in the format
        ``name@mapset|start_time[|end_time]|semantic_label``
    :param str output: Name of the output STRDS
    :param SpaceTimeRasterDataset input_stds: Input STRDS the output is
        derived from, used for initial metadata (temporal type, semantic type,
        title)
    :param bool extend: Whether to extend an existing output STRDS instead
        of recreating it
    :param bool overwrite: Whether to overwrite a existing output
    """
    tools = Tools()
    output_id = f"{output}@{tgis.get_current_mapset()}"
    output_stds = tgis.dataset_factory("strds", output_id)
    output_stds_exists = output_stds.is_in_db()
    recreate = False
    if output_stds_exists and not extend:
        tgis.check_new_stds(output, "strds", None, overwrite)
        recreate = True
    if not output_stds_exists or recreate:
        temporal_type, semantic_type, title, _description = (
            input_stds.get_initial_values()
        )
        tools.t_create(
            output=output,
            type="strds",
            title=f"Wet Snow computed oriented on Nagler & Rott 2000 from {title}",
            description=f"Wet Snow computed oriented on Nagler & Rott 2000 from {title}",
            temporaltype=temporal_type,
            semantictype=semantic_type,
            overwrite=overwrite,
            quiet=True,
        )

    with NamedTemporaryFile("wt", encoding="utf-8") as register_file:
        register_file.write("".join(map_list))
        tools.t_register(
            input=output_id,
            file=register_file.name,
            overwrite=overwrite,
            quiet=True,
        )


def main() -> None:
    """Fetch data and compute wetSnow."""
    overwrite = gs.overwrite()  # True
    if flags["e"] and not overwrite:
        gs.fatal(_("Option -e requires overwrite to be enabled."))

    # Check options
    basename = options["basename"]  # "Sentinel_1_WetSnow",
    method = options["method"]  # "median"
    nprocs = int(options["nprocs"]) or os.cpu_count() or 1
    median_filter = int(options["size"])  # 5
    mode_filter = int(options["mode_size"])  # 3
    for label, value in (("size", median_filter), ("mode_size", mode_filter)):
        if value != 0 and value % 2 == 0:
            gs.fatal(_("Option <%s> must be odd, got %i.") % (label, value))
    k = float(options["k"])  # 0.5
    detection_thresholds = sorted(
        map(
            float,
            options["detection_thresholds"].split(","),
        ),
    )
    upper_detection_threshold = max(detection_thresholds)
    lower_detection_threshold = min(detection_thresholds)
    nodata_threshold = float(options["nodata_threshold"])
    memory = int(options["memory"])

    # Initialize TGIS
    tgis.init()

    # Assumes input STDS has semantic labels that identify polarization and track:
    # with semantic_labels like e.g.: VV_dbi_110_descending
    input_stds = tgis.open_old_stds(options["input"], "strds")
    raster_maps = group_input_maps(
        input_stds.get_registered_maps(where=options["where"]),
    )

    # Drop time steps with missing polarization, and tracks left with none
    raster_maps = {
        track: complete_extents
        for track, temporal_extents in raster_maps.items()
        if (complete_extents := check_complete_polarizations(temporal_extents))
    }

    reference_maps = get_reference_data(options["reference_pattern"])

    missing_reference = sorted(set(raster_maps).difference(set(reference_maps)))
    if missing_reference:
        gs.warning(
            _("No reference data found for tracks:\n%s\nSkipping...")
            % ", ".join(map(str, missing_reference)),
        )

    tracks_to_process = set(raster_maps).intersection(set(reference_maps))
    if not tracks_to_process:
        gs.warning(_("No tracks found to process."))
        return

    # Keep only the tracks that will actually be processed
    for track in list(raster_maps):
        if track not in tracks_to_process:
            del raster_maps[track]

    # Flatten the nested raster_maps dict into fully-bound, zero-arg tasks
    tasks = [
        partial(
            compute_wet_snow,
            track,
            temporal_extent,
            raster_map_dict["VV"],
            raster_map_dict["VH"],
            reference_maps[track],
            basename=basename,
            nprocs=1,
            memory=memory,
            method=method,
            median_filter=median_filter,
            mode_filter=mode_filter,
            k=k,
            upper_detection_threshold=upper_detection_threshold,
            lower_detection_threshold=lower_detection_threshold,
            nodata_threshold=nodata_threshold,
            overwrite=overwrite,
            temp_name=f"{TEMP_NAME}_{track}_{temporal_extent[0].strftime('%Y%m%d%H%M%S')}",
            mapset=gs.gisenv()["MAPSET"],
        )
        for track, temporal_extents in raster_maps.items()
        for temporal_extent, raster_map_dict in temporal_extents.items()
    ]

    wet_snow_maps = []
    with Pool(processes=nprocs) as pool:
        for idx, result in enumerate(pool.imap_unordered(call, tasks)):
            gs.percent(idx, len(tasks), 3)
            wet_snow_maps.append(result)
    cleanup()

    register_in_tgis(
        wet_snow_maps,
        options["output"],
        input_stds,
        extend=flags["e"],
        overwrite=overwrite,
    )


if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    # Lazy-import TGIS
    import grass.temporal as tgis

    main()
