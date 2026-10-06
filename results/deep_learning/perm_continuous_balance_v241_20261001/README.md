# perm_continuous_balance_v241

## Focus
Matched perm_probability_mass_v230 architecture/init/optimizer/seed. Only separately normalize continuous likelihood to prevent majority default rows diluting training signal. Q values remain unchanged; minibatch training labels define denominator.

## Explored and validation
Prepared, not launched. Same original parent, zero repair output, no previous trained repair weights. Own-target checkpoint; final all144665 assembly and frozen/RAM restored replay.

## Gain
Unproved. Max80/patience14/min10. No validation-derived weighting. Max2jobs; individual and batch wallclock recorded separately after completion.

## Not worth continuing
No conclusion yet. If normalization does not help, close this particular balance adjustment rather than an endless coefficient sweep. Existing positive recipe preserved.

V241 final PERM89.413463686933, gain+0.007898830014, actualTOTAL80.615240008499; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds292.098263/peakMiB791.250000. No otherfolds.

241 final matched PERM loss normalization89.413463686933 vs23089.413121051402,+0.000342635531; best10/stop24. Actualoptional7TOTAL80.624489319708, gain+0.000119922436, same144665fifthpopulation. Tiny positive retained; default5 unchanged. SW242 currently training87424params(74848existingFF+12576mass),peak976.59MiB; no proven gain. Historical interpolation CUDAbackward warn-only nondeterminism observed whenFF enabled; no deterministic bitwise rerun claim. No otherfolds/fullOOF proof.
