I built a U-Net to beat the standard burned-area index. On the biome it never
saw, the index won.

burned-area-s2 segments burn scars from Sentinel-2 pre/post pairs on three
Copernicus EMS fires from the 2026 French season. Train on Saumos (31,602 ha,
pine), test on Biscarrosse (1,753 ha, pine) and Fontainebleau (832 ha,
broadleaf), so a size effect can be told apart from a biome effect.

Both methods get exactly one decision parameter, calibrated by the same
procedure on the same 772,271 held-out pixels and frozen before test. dNBR at
0.576, the network at 0.885.

Then a third row. The oracle refits the index's threshold on the test event
itself, which is impossible in production since it needs the answer to produce
the answer. But it is the ceiling of what the index can reach at any threshold,
so it splits the network's lead in two: the part a cheap local recalibration
would also have bought, and the part that is genuine spatial context.

On pine, the network beat that ceiling. IoU 0.863 vs 0.812 on Saumos, 0.571 vs
0.526 on Biscarrosse. No threshold on the index reaches those numbers, and a
1:23 change of scale didn't touch the effect.

On broadleaf, it didn't. Against the honest baseline, Fontainebleau looks like
the network's best result: IoU 0.556 against 0.192, an index missing 80% of the
burned area. The oracle reaches 0.624. So the network's entire lead over the
honest baseline on the unseen biome was calibration transfer, and one
recalibrated threshold would have bought more of it, for the price of one
number. Threshold-free average precision says the same from the other side:
0.839 for the index, 0.830 for the network.

On this evidence, deploying this network to a new biome isn't justified.
Recalibrating the index there is. That was not the answer I set the project up
hoping for.

Two more things I'd rather report than hide.

No confidence intervals are published anywhere. The block bootstrap counts
blocks that actually contain burned pixels, and both transfer events fit inside
a single one. Every point estimate stands alone and carries its reason in the
results JSON. A declared hole beats a false interval.

And the first baseline numbers I published were wrong. The Sentinel-2
baseline-04.00 offset was applied twice, once by the provider and once by my
code, which crippled the index while leaving the network untouched because
per-channel standardisation absorbs an additive shift. That is the exact shape
of an unfair comparison, and I only found it while checking the network's
inputs. Fixed, whole chain re-run, Saumos dNBR IoU 0.521 to 0.702. A guard now
refuses any band whose valid pixels are more than 10% negative.

Still to come: the error maps that would confirm the clearcut hypothesis behind
the pine result, and a few-shot adaptation curve.

It all runs on CPU. 40 epochs of a 24.5M-parameter U-Net take about ten minutes,
on public COGs, no credentials.

Repo, results table and figures: [link]
