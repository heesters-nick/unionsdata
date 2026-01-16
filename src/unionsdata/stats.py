"""Statistics tracking and reporting for unionsdata pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from rich.console import Console
from rich.table import Table

from unionsdata.utils import TileAvailability

logger = logging.getLogger(__name__)


@dataclass
class BandAvailability:
    """Track availability of data for a single band."""

    requested: int = 0
    available: int = 0

    @property
    def unavailable(self) -> int:
        return self.requested - self.available


@dataclass
class BandProcessingStats:
    """Track processing outcomes for a single band."""

    succeeded: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def total_processed(self) -> int:
        return self.succeeded + self.skipped + self.failed


@dataclass
class RunStatistics:
    """Unified statistics for a pipeline run."""

    bands: list[str] = field(default_factory=list)

    # Availability (pre-processing)
    tile_availability: dict[str, BandAvailability] = field(default_factory=dict)
    cutout_availability: dict[str, BandAvailability] = field(default_factory=dict)

    # Processing outcomes (post-processing)
    tile_processing: dict[str, BandProcessingStats] = field(default_factory=dict)
    cutout_processing: dict[str, BandProcessingStats] = field(default_factory=dict)

    def initialize_bands(self, bands: list[str]) -> None:
        """Initialize tracking structures for the given bands."""
        self.bands = bands
        for band in bands:
            self.tile_availability[band] = BandAvailability()
            self.cutout_availability[band] = BandAvailability()
            self.tile_processing[band] = BandProcessingStats()
            self.cutout_processing[band] = BandProcessingStats()

    def record_tile_download(self, band: str, status: str) -> None:
        """Record a tile download outcome."""
        if band not in self.tile_processing:
            self.tile_processing[band] = BandProcessingStats()

        stats = self.tile_processing[band]
        if status == 'downloaded':
            stats.succeeded += 1
        elif status == 'skipped':
            stats.skipped += 1
        elif status == 'failed':
            stats.failed += 1

    def record_cutout_result(
        self,
        band: str,
        succeeded: int = 0,
        skipped: int = 0,
        failed: int = 0,
    ) -> None:
        """Record cutout processing outcomes for a band."""
        if band not in self.cutout_processing:
            self.cutout_processing[band] = BandProcessingStats()

        stats = self.cutout_processing[band]
        stats.succeeded += succeeded
        stats.skipped += skipped
        stats.failed += failed


def compute_tile_availability(
    download_jobs: list[tuple[tuple[int, int], str]],
    bands: list[str],
    stats: RunStatistics,
) -> None:
    """
    Compute tile availability from download jobs.

    Since download_jobs only contains (tile, band) pairs for tiles that exist
    on the server, we track what was actually queued for download.

    Args:
        download_jobs: List of (tile, band) tuples to download
        bands: List of requested bands
        stats: RunStatistics object to update
    """
    # Count jobs per band
    jobs_per_band: dict[str, int] = dict.fromkeys(bands, 0)
    for _, band in download_jobs:
        if band in jobs_per_band:
            jobs_per_band[band] += 1

    for band in bands:
        stats.tile_availability[band].available = jobs_per_band[band]


def compute_tile_availability_from_catalog(
    catalog: pd.DataFrame,
    bands: list[str],
    stats: RunStatistics,
) -> None:
    """
    Compute tile availability based on catalog objects and their tile assignments.

    For object-based input (coordinates/table), we count unique tiles per band
    that the objects fall into.

    Args:
        catalog: Augmented catalog with 'tile' and 'bands_available' columns
        bands: List of requested bands
        stats: RunStatistics object to update
    """
    if catalog.empty:
        return

    # Get unique tiles from the catalog
    unique_tiles = catalog['tile'].unique()
    n_tiles_requested = len(unique_tiles)

    for band in bands:
        stats.tile_availability[band].requested = n_tiles_requested

        # Count tiles that have this band available
        # A tile has the band if ANY object in that tile has it in bands_available
        tiles_with_band = set()
        for tile in unique_tiles:
            tile_rows = catalog[catalog['tile'] == tile]
            for bands_avail in tile_rows['bands_available']:
                if pd.notna(bands_avail) and band in str(bands_avail).split(','):
                    tiles_with_band.add(tile)
                    break

        stats.tile_availability[band].available = len(tiles_with_band)


def compute_cutout_availability(
    catalog: pd.DataFrame,
    bands: list[str],
    stats: RunStatistics,
) -> None:
    """
    Compute cutout availability from catalog.

    Args:
        catalog: Augmented catalog with 'bands_available' column
        bands: List of requested bands
        stats: RunStatistics object to update
    """
    if catalog.empty:
        return

    n_objects = len(catalog)

    for band in bands:
        stats.cutout_availability[band].requested = n_objects

        # Count objects where this band is available
        available_count = 0
        for bands_avail in catalog['bands_available']:
            if pd.notna(bands_avail) and band in str(bands_avail).split(','):
                available_count += 1

        stats.cutout_availability[band].available = available_count


def compute_tile_availability_from_tiles(
    requested_tiles: list[tuple[int, int]],
    bands: list[str],
    avail: object,  # TileAvailability - avoid circular import
    stats: RunStatistics,
) -> None:
    """
    Compute tile availability for tile-based input.

    Args:
        requested_tiles: List of tile tuples requested by user
        bands: List of requested bands
        avail: TileAvailability object
        stats: RunStatistics object to update
    """
    if not isinstance(avail, TileAvailability):
        return

    n_tiles_requested = len(requested_tiles)
    requested_set = set(requested_tiles)

    for band in bands:
        stats.tile_availability[band].requested = n_tiles_requested

        # Get tiles available in this band and intersect with requested
        band_tiles = set(avail.band_tiles(band))
        available_tiles = requested_set & band_tiles

        stats.tile_availability[band].available = len(available_tiles)


def compute_cutout_skipped(
    catalog: pd.DataFrame,
    bands: list[str],
    stats: RunStatistics,
) -> None:
    """
    Compute skipped cutout counts from catalog's existing cutout_bands.

    Call this BEFORE processing starts to record what's already done locally.
    """
    if catalog.empty or 'cutout_bands' not in catalog.columns:
        return

    for band in bands:
        skipped_count = 0
        for cutout_bands_str in catalog['cutout_bands']:
            if pd.notna(cutout_bands_str) and cutout_bands_str:
                existing_bands = set(str(cutout_bands_str).split(','))
                existing_bands.discard('')
                if band in existing_bands:
                    skipped_count += 1
        stats.cutout_processing[band].skipped = skipped_count


def report_summary(
    stats: RunStatistics,
    cutouts_enabled: bool,
    cutout_mode: str,
    show_tiles: bool = True,
) -> None:
    """
    Print formatted summary tables using rich.

    Args:
        stats: RunStatistics object with all collected data
        cutouts_enabled: Whether cutout processing was enabled
        cutout_mode: Mode of cutout operation
        show_tiles: Whether to show tile statistics (False for direct_only mode)
    """
    console = Console()

    # Determine if we should show tile stats
    show_tile_stats = show_tiles and cutout_mode != 'direct_only'

    # Check if we have any data to show
    has_tile_availability = any(a.requested > 0 for a in stats.tile_availability.values())
    has_cutout_availability = any(a.requested > 0 for a in stats.cutout_availability.values())
    has_tile_processing = any(p.total_processed > 0 for p in stats.tile_processing.values())
    has_cutout_processing = any(p.total_processed > 0 for p in stats.cutout_processing.values())

    # ==================== AVAILABILITY SECTION ====================
    console.print()
    console.print('=' * 60, style='bold')
    console.print('AVAILABILITY', style='bold blue', justify='left')
    console.print('=' * 60, style='bold')

    # Tile availability table
    if show_tile_stats and has_tile_availability:
        console.print()
        console.print('📊 TILE AVAILABILITY BY BAND:', style='bold')
        console.print()

        tile_avail_table = Table(show_header=True, header_style='bold cyan')
        tile_avail_table.add_column('Band', style='white')
        tile_avail_table.add_column('Requested', justify='right')
        tile_avail_table.add_column('Available', justify='right', style='green')
        tile_avail_table.add_column('Unavailable', justify='right', style='yellow')

        total_requested = 0
        total_available = 0

        for band in stats.bands:
            avail = stats.tile_availability.get(band, BandAvailability())
            total_requested += avail.requested
            total_available += avail.available

            unavail_style = 'red' if avail.unavailable > 0 else 'green'
            tile_avail_table.add_row(
                band,
                str(avail.requested),
                str(avail.available),
                f'[{unavail_style}]{avail.unavailable}[/{unavail_style}]',
            )

        # Add totals row
        total_unavailable = total_requested - total_available
        unavail_style = 'red' if total_unavailable > 0 else 'green'
        tile_avail_table.add_section()
        tile_avail_table.add_row(
            '[bold]TOTAL[/bold]',
            f'[bold]{total_requested}[/bold]',
            f'[bold green]{total_available}[/bold green]',
            f'[bold {unavail_style}]{total_unavailable}[/bold {unavail_style}]',
        )

        console.print(tile_avail_table)

    # Cutout availability table
    if cutouts_enabled and has_cutout_availability:
        console.print()
        console.print('📊 CUTOUT AVAILABILITY BY BAND:', style='bold')
        console.print()

        cutout_avail_table = Table(show_header=True, header_style='bold cyan')
        cutout_avail_table.add_column('Band', style='white')
        cutout_avail_table.add_column('Requested', justify='right')
        cutout_avail_table.add_column('Available', justify='right', style='green')
        cutout_avail_table.add_column('Unavailable', justify='right', style='yellow')

        total_requested = 0
        total_available = 0

        for band in stats.bands:
            avail = stats.cutout_availability.get(band, BandAvailability())
            total_requested += avail.requested
            total_available += avail.available

            unavail_style = 'red' if avail.unavailable > 0 else 'green'
            cutout_avail_table.add_row(
                band,
                str(avail.requested),
                str(avail.available),
                f'[{unavail_style}]{avail.unavailable}[/{unavail_style}]',
            )

        # Add totals row
        total_unavailable = total_requested - total_available
        unavail_style = 'red' if total_unavailable > 0 else 'green'
        cutout_avail_table.add_section()
        cutout_avail_table.add_row(
            '[bold]TOTAL[/bold]',
            f'[bold]{total_requested}[/bold]',
            f'[bold green]{total_available}[/bold green]',
            f'[bold {unavail_style}]{total_unavailable}[/bold {unavail_style}]',
        )

        console.print(cutout_avail_table)

    # ==================== PROCESSING SECTION ====================
    console.print()
    console.print('=' * 60, style='bold')
    console.print('PROCESSING RESULTS', style='bold blue', justify='left')
    console.print('=' * 60, style='bold')

    # Tile processing table
    if show_tile_stats and has_tile_processing:
        console.print()
        console.print('⬇️  TILE DOWNLOADS BY BAND:', style='bold')
        console.print()

        tile_proc_table = Table(show_header=True, header_style='bold cyan')
        tile_proc_table.add_column('Band', style='white')
        tile_proc_table.add_column('Succeeded', justify='right', style='green')
        tile_proc_table.add_column('Skipped', justify='right', style='yellow')
        tile_proc_table.add_column('Failed', justify='right', style='red')

        total_succeeded = 0
        total_skipped = 0
        total_failed = 0

        for band in stats.bands:
            proc = stats.tile_processing.get(band, BandProcessingStats())
            total_succeeded += proc.succeeded
            total_skipped += proc.skipped
            total_failed += proc.failed

            failed_style = 'red' if proc.failed > 0 else 'dim'
            tile_proc_table.add_row(
                band,
                str(proc.succeeded),
                str(proc.skipped),
                f'[{failed_style}]{proc.failed}[/{failed_style}]',
            )

        # Add totals row
        failed_style = 'red' if total_failed > 0 else 'dim'
        tile_proc_table.add_section()
        tile_proc_table.add_row(
            '[bold]TOTAL[/bold]',
            f'[bold green]{total_succeeded}[/bold green]',
            f'[bold yellow]{total_skipped}[/bold yellow]',
            f'[bold {failed_style}]{total_failed}[/bold {failed_style}]',
        )

        console.print(tile_proc_table)

    # Cutout processing table
    if cutouts_enabled and has_cutout_processing:
        console.print()
        console.print('✂️  CUTOUTS BY BAND:', style='bold')
        console.print()

        cutout_proc_table = Table(show_header=True, header_style='bold cyan')
        cutout_proc_table.add_column('Band', style='white')
        cutout_proc_table.add_column('Succeeded', justify='right', style='green')
        cutout_proc_table.add_column('Skipped', justify='right', style='yellow')
        cutout_proc_table.add_column('Failed', justify='right', style='red')

        total_succeeded = 0
        total_skipped = 0
        total_failed = 0

        for band in stats.bands:
            proc = stats.cutout_processing.get(band, BandProcessingStats())
            total_succeeded += proc.succeeded
            total_skipped += proc.skipped
            total_failed += proc.failed

            failed_style = 'red' if proc.failed > 0 else 'dim'
            cutout_proc_table.add_row(
                band,
                str(proc.succeeded),
                str(proc.skipped),
                f'[{failed_style}]{proc.failed}[/{failed_style}]',
            )

        # Add totals row
        failed_style = 'red' if total_failed > 0 else 'dim'
        cutout_proc_table.add_section()
        cutout_proc_table.add_row(
            '[bold]TOTAL[/bold]',
            f'[bold green]{total_succeeded}[/bold green]',
            f'[bold yellow]{total_skipped}[/bold yellow]',
            f'[bold {failed_style}]{total_failed}[/bold {failed_style}]',
        )

        console.print(cutout_proc_table)

    console.print()
    console.print('=' * 60, style='bold')
    console.print()
