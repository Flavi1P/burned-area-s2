# burned-area-s2

Burned-area segmentation from Sentinel-2 pre/post pairs: a U-Net compared
against a thresholded dNBR baseline, on three Copernicus EMS fires from the
2026 French season.

The aim is to locate *where* a per-pixel index fails, and to see whether
spatial context fixes those specific failures. So both methods get a single
decision threshold, calibrated the same way on the same held-out pixels and
frozen before test, and the split is by event: train on Saumos (31 602 ha,
pine), test on Biscarrosse (1 753 ha, pine) and Fontainebleau (832 ha,
broadleaf). Comparing the two drops separates a size effect from a biome
effect.

**What works today:** scene search and download from the public Sentinel-2
archive, EMS label rasterisation on the 20 m grid, and the figures below.
**What comes next:** the dNBR baseline and its metrics, then the U-Net.
No accuracy numbers are published yet.

![Saumos before / after / EMS label](outputs/quicklook_saumos.png)

![Fontainebleau before / after / EMS label](outputs/quicklook_fontainebleau.png)

False colour B12/B8A/B04 at 20 m — three of the four bands the model sees.
Scars read dark maroon.

## Running it

```bash
conda env create -f environment.yml
conda activate burned-area-s2

python -m src.config                              # events and settings, as loaded
python -m src.data.ems                            # EMS label provenance
python -m src.data.stac --event saumos --list     # candidate scenes + cloud
python -m src.viz.quicklook --event saumos        # before / after / label
pytest -q
```

No credentials needed. The imagery is public COGs from the Element 84 STAC
catalogue.

## Layout

```
config.yaml     events, experiments, evaluation settings. No event name,
                EMS code or date lives anywhere else.
src/data/       STAC access, EMS labels, target grids
src/model/      dataset, U-Net, training loop
src/eval/       dNBR, thresholds, metrics, blocks, bootstrap
src/viz/        figures
outputs/        versioned figures and tables, never rasters
tests/          guards on the evaluation setup
```

The tests exist because a leak produces a plausible number rather than a
crash, and that's the harder failure to catch: a test event can't appear in
training, the threshold can't be calibrated on a test event, test tiles can't
overlap, negative test tiles can't be filtered out.

## Licence

MIT. Copernicus EMS products and Sentinel-2 data under their respective terms.
