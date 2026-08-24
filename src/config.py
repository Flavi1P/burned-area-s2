"""Configuration loading and validation.

The experimental plan is data, not code. Everything downstream resolves an
event or an experiment by identifier through this module; nothing anywhere else
in the repository is allowed to name an event, an EMS activation or a date.

The validation here is deliberately opinionated: it refuses configurations that
would silently break the evaluation protocol (an experiment testing on an event
it trained on, a threshold calibrated on a test event, a forbidden metric).
Those are the failure modes that produce a plausible-looking number nobody can
defend afterwards.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when config.yaml is internally inconsistent."""


# --------------------------------------------------------------------------- #
# leaf records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabelProduction:
    """How the EMS analyst actually produced the delineation.

    Section 5 of the design note: EMS production is not homogeneous. A
    semi-automatic, index-derived delineation makes "U-Net beats dNBR" partly
    circular, so the method is recorded per event and reported in the README.
    Populated from the ISO 19115 lineage by ``src.data.ems``.
    """

    method: str | None = None
    delineated_on: str | None = None   # the post-event image actually digitised
    analysis_scale: str | None = None
    mmu_m2: float | None = None
    geometric_rmse_m: float | None = None

    @property
    def is_documented(self) -> bool:
        return self.method is not None


@dataclass(frozen=True)
class Label:
    activation: str
    aoi: str
    product: str
    version: int
    status: str
    package_url: str
    reference_date: dt.date
    source_imagery: tuple[str, ...]
    production: LabelProduction
    reported_burnt_area_ha: float | None = None
    alternative_products: tuple[dict[str, Any], ...] = ()

    @property
    def product_id(self) -> str:
        """e.g. EMSRnnn_AOI01_DEL_MONITnn_v1 -- the unit of provenance."""
        return f"{self.activation}_{self.aoi}_{self.product}_v{self.version}"

    @property
    def is_final(self) -> bool:
        return self.status.lower() == "final"


@dataclass(frozen=True)
class Imagery:
    pre_window: tuple[dt.date, dt.date]
    post_window: tuple[dt.date, dt.date]
    pre_date: dt.date | None = None
    post_date: dt.date | None = None


@dataclass(frozen=True)
class Event:
    id: str
    name: str
    admin: str
    fuel: str
    role: str
    event_datetime: dt.datetime
    aoi_bbox: tuple[float, float, float, float]
    label: Label
    imagery: Imagery
    aoi_span_km: tuple[float, float] | None = None
    test_purpose: str | None = None
    press_area_ha: float | None = None

    @property
    def is_train(self) -> bool:
        return self.role == "train"


@dataclass(frozen=True)
class Experiment:
    id: str
    question: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    split: str
    finetune_on: str | None = None
    finetune_fractions: tuple[float, ...] = ()
    n_partitions: int | None = None


@dataclass(frozen=True)
class Config:
    """Parsed config.yaml.

    ``raw`` keeps the untouched mapping so that sections which are only
    consumed later (model hyper-parameters, evaluation knobs) do not need a
    dataclass before the phase that uses them.
    """

    raw: dict[str, Any]
    path: Path
    events: dict[str, Event]
    experiments: dict[str, Experiment]

    # -- convenience accessors ------------------------------------------- #

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def stac(self) -> dict[str, Any]:
        return self.raw["stac"]

    @property
    def ems(self) -> dict[str, Any]:
        return self.raw["ems"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def pixel_model(self) -> dict[str, Any]:
        return self.raw["pixel_model"]

    @property
    def band_assets(self) -> list[str]:
        """STAC asset keys of the four reflectance bands, in channel order."""
        return list(self.project["bands"].keys())

    @property
    def resolution_m(self) -> float:
        return float(self.project["target_resolution_m"])

    def event(self, event_id: str) -> Event:
        try:
            return self.events[event_id]
        except KeyError:
            known = ", ".join(sorted(self.events))
            raise ConfigError(
                f"unknown event {event_id!r}; config.yaml declares: {known}"
            ) from None

    def experiment(self, experiment_id: str) -> Experiment:
        try:
            return self.experiments[experiment_id]
        except KeyError:
            known = ", ".join(sorted(self.experiments))
            raise ConfigError(
                f"unknown experiment {experiment_id!r}; config.yaml declares: {known}"
            ) from None

    def path_for(self, key: str, *parts: str) -> Path:
        """Resolve a configured directory, relative to the repository root."""
        base = REPO_ROOT / self.raw["paths"][key]
        return base.joinpath(*parts)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _as_date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ConfigError(f"{where}: expected a date, got {value!r}")


def _as_datetime(value: Any, where: str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value)
    raise ConfigError(f"{where}: expected a datetime, got {value!r}")


def _window(value: Any, where: str) -> tuple[dt.date, dt.date]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{where}: expected a [start, end] pair, got {value!r}")
    start, end = (_as_date(v, where) for v in value)
    if end < start:
        raise ConfigError(f"{where}: window ends ({end}) before it starts ({start})")
    return start, end


def _parse_event(d: dict[str, Any]) -> Event:
    eid = d["id"]
    lab = d["label"]
    prod = lab.get("production") or {}

    bbox = d["aoi_bbox"]
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ConfigError(f"event {eid}: aoi_bbox must be [minx, miny, maxx, maxy]")

    label = Label(
        activation=lab["activation"],
        aoi=lab["aoi"],
        product=lab["product"],
        version=int(lab["version"]),
        status=lab["status"],
        package_url=lab["package_url"],
        reference_date=_as_date(lab["reference_date"], f"event {eid}: reference_date"),
        source_imagery=tuple(lab.get("source_imagery") or ()),
        production=LabelProduction(
            method=prod.get("method"),
            delineated_on=prod.get("delineated_on"),
            analysis_scale=prod.get("analysis_scale"),
            mmu_m2=prod.get("mmu_m2"),
            geometric_rmse_m=prod.get("geometric_rmse_m"),
        ),
        reported_burnt_area_ha=lab.get("reported_burnt_area_ha"),
        alternative_products=tuple(lab.get("alternative_products") or ()),
    )

    img = d["imagery"]
    imagery = Imagery(
        pre_window=_window(img["pre_window"], f"event {eid}: pre_window"),
        post_window=_window(img["post_window"], f"event {eid}: post_window"),
        pre_date=_as_date(img["pre_date"], f"event {eid}: pre_date")
        if img.get("pre_date")
        else None,
        post_date=_as_date(img["post_date"], f"event {eid}: post_date")
        if img.get("post_date")
        else None,
    )

    span = d.get("aoi_span_km")
    return Event(
        id=eid,
        name=d["name"],
        admin=d["admin"],
        fuel=d["fuel"],
        role=d["role"],
        event_datetime=_as_datetime(d["event_datetime"], f"event {eid}"),
        aoi_bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
        label=label,
        imagery=imagery,
        aoi_span_km=tuple(float(v) for v in span) if span else None,  # type: ignore[arg-type]
        test_purpose=d.get("test_purpose"),
        press_area_ha=d.get("press_area_ha"),
    )


def _parse_experiment(d: dict[str, Any]) -> Experiment:
    return Experiment(
        id=d["id"],
        question=" ".join(str(d.get("question", "")).split()),
        train=tuple(d.get("train") or ()),
        test=tuple(d.get("test") or ()),
        split=d["split"],
        finetune_on=d.get("finetune_on"),
        finetune_fractions=tuple(float(f) for f in d.get("finetune_fractions") or ()),
        n_partitions=d.get("n_partitions"),
    )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _validate(cfg: Config) -> None:
    events, experiments = cfg.events, cfg.experiments

    if not events:
        raise ConfigError("config.yaml declares no events")

    train_events = {e.id for e in events.values() if e.is_train}
    if not train_events:
        raise ConfigError("no event has role 'train'")

    for exp in experiments.values():
        for role, ids in (("train", exp.train), ("test", exp.test)):
            for eid in ids:
                if eid not in events:
                    raise ConfigError(
                        f"experiment {exp.id}: {role} references unknown event {eid!r}"
                    )
        if exp.finetune_on and exp.finetune_on not in events:
            raise ConfigError(
                f"experiment {exp.id}: finetune_on references unknown event "
                f"{exp.finetune_on!r}"
            )

        # An experiment may legitimately train and test on the same event (E1),
        # but only through spatially disjoint blocks. Sharing an event under a
        # by_event split is the classic leak.
        shared = set(exp.train) & set(exp.test)
        if shared and exp.split != "spatial_blocks":
            raise ConfigError(
                f"experiment {exp.id}: events {sorted(shared)} are in both train and "
                f"test under split={exp.split!r}; only 'spatial_blocks' may do that"
            )

        # E3 fine-tunes on the event it is evaluated on: same rule.
        if exp.finetune_on and exp.finetune_on in exp.test:
            if exp.split != "spatial_blocks":
                raise ConfigError(
                    f"experiment {exp.id}: fine-tunes and tests on "
                    f"{exp.finetune_on!r} without a spatial_blocks split"
                )
            if not exp.n_partitions or exp.n_partitions < 3:
                raise ConfigError(
                    f"experiment {exp.id}: fine-tuning and testing on the same event "
                    "needs at least 3 alternating partitions, so the curve carries a "
                    "spread rather than one lucky split"
                )

    # The decision threshold must never see a test event -- that is a leak on
    # both sides of the comparison, and it is the leak nobody notices.
    calib = cfg.evaluation["threshold"]["calibrated_on"]
    if calib not in events:
        raise ConfigError(f"evaluation.threshold.calibrated_on: unknown event {calib!r}")
    for exp in experiments.values():
        if calib in exp.test and exp.split != "spatial_blocks":
            raise ConfigError(
                f"evaluation.threshold.calibrated_on={calib!r} is a test event of "
                f"{exp.id} under split={exp.split!r}: the threshold would be fitted "
                "on evaluation pixels"
            )
    if cfg.evaluation["threshold"].get("recalibrate_on_target"):
        raise ConfigError(
            "evaluation.threshold.recalibrate_on_target must stay false: recalibrating "
            "on the target is exactly the transfer cost E2 is meant to measure"
        )

    forbidden = set(cfg.evaluation.get("forbidden_metrics") or ())
    reported = set(cfg.evaluation.get("metrics") or ())
    if forbidden & reported:
        raise ConfigError(
            f"evaluation.metrics contains forbidden metrics: {sorted(forbidden & reported)}"
        )

    if cfg.evaluation["tiling"]["test"].get("overlap_px", 0) != 0:
        raise ConfigError(
            "evaluation.tiling.test.overlap_px must be 0: overlapping test tiles "
            "manufacture dependence between evaluation units and invalidate the "
            "block bootstrap"
        )
    if cfg.evaluation["tiling"]["test"].get("filter_negatives", False):
        raise ConfigError(
            "evaluation.tiling.test.filter_negatives must be false: dropping negative "
            "tiles inflates precision by an arbitrary factor"
        )

    tile_px = int(cfg.project["tile_size_px"])
    tile_km = tile_px * cfg.resolution_m / 1000.0
    block_km = float(cfg.evaluation["spatial_blocks"]["superblock_km"])
    if block_km <= tile_km:
        raise ConfigError(
            f"evaluation.spatial_blocks.superblock_km ({block_km} km) must exceed the "
            f"tile size ({tile_km} km); using the tile as bootstrap unit repeats the "
            "spatial leak in a subtler form"
        )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate config.yaml."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    events = {}
    for d in raw.get("events") or []:
        ev = _parse_event(d)
        if ev.id in events:
            raise ConfigError(f"duplicate event id {ev.id!r}")
        events[ev.id] = ev

    experiments = {}
    for d in raw.get("experiments") or []:
        exp = _parse_experiment(d)
        if exp.id in experiments:
            raise ConfigError(f"duplicate experiment id {exp.id!r}")
        experiments[exp.id] = exp

    cfg = Config(raw=raw, path=path, events=events, experiments=experiments)
    _validate(cfg)
    return cfg


def _describe(cfg: Config) -> str:
    lines = [f"{cfg.path}", ""]
    lines.append("events")
    for ev in cfg.events.values():
        prod = ev.label.production
        method = prod.method or "not yet read from EMS metadata"
        lines.append(
            f"  {ev.id:<15} {ev.role:<5} {ev.label.product_id:<34} "
            f"{ev.label.status:<10} {ev.label.reported_burnt_area_ha:>10} ha"
        )
        lines.append(f"  {'':<15} label method: {method}")
    lines.append("")
    lines.append("experiments")
    for exp in cfg.experiments.values():
        lines.append(
            f"  {exp.id:<4} train={list(exp.train)} test={list(exp.test)} "
            f"split={exp.split}"
        )
        lines.append(f"       {exp.question}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(_describe(load_config()))
