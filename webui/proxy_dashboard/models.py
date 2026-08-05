"""Persisted price-proxy / Hydrogen Producer runs.

Unlike Project 3's ``dispatcher`` app (which persists nothing — every page is a
fresh compute), this app keeps one row per (country, run) so a previously-run
country's results can be redisplayed instantly without re-solving anything --
see ``views.country_detail``.
"""
from django.db import models


class RunResult(models.Model):
    """``elec_metrics``/``h2_metrics``: ``{"r2": .., "rmse": .., "mae": .., "corr": ..}``
    or null if no elec/h2 zone. ``price_series``: ``{"hours": [...], "elec_real": [...],
    "elec_pred": [...], "h2_real": [...], "h2_pred": [...]}``. ``exchange``:
    ``{"grid_mwh": {"ours": .., "actual": ..}, "pipeline_mwh": {...}, "gc_bought_mwh":
    {...}, "gc_sold_mwh": {...}, "objective_eur": .., "solve_seconds": ..}`` or null if
    the LP wasn't run/failed."""
    country = models.CharField(max_length=2, db_index=True)
    elec_zone = models.CharField(max_length=8, null=True, blank=True)
    h2_zone = models.CharField(max_length=8, null=True, blank=True)
    start_day = models.IntegerField()
    end_day = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    elec_metrics = models.JSONField(null=True, blank=True)
    h2_metrics = models.JSONField(null=True, blank=True)

    price_series = models.JSONField(default=dict, blank=True)

    exchange = models.JSONField(null=True, blank=True)
    lp_error = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['country', '-created_at'])]

    def __str__(self):
        return f"{self.country} run @ {self.created_at:%Y-%m-%d %H:%M} (days {self.start_day}-{self.end_day})"
