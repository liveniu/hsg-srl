# Matching-neutrality probe, stage A

Released clock maps retained. Tracks re-assigned with the released matcher functions and gates, once with HSG coordinates (reproduction check) and once with fixed-midpoint coordinates. Test split, stored footprint-center references. Intervals are 1,000 camera/track cluster bootstrap draws for median(HSG) - median(midpoint) and the RMSE difference.

| Recording | Released test matched | Midpoint matched | Common | Identity changed | Only released | Only midpoint | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 184,442 | 185,297 | 183,935 | 0 | 507 | 1,362 | 0.9899 |
| M6 | 134,382 | 134,465 | 134,084 | 0 | 298 | 381 | 0.9950 |
| Ko-PER | 25,122 | 25,122 | 25,122 | 0 | 0 | 0 | 1.0000 |

| Recording | Set | n | HSG median / RMSE / p90 | Midpoint median / RMSE / p90 | >2 m HSG / midpoint | median diff [interval] | RMSE diff [interval] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | released_hsg_matching | 184,442 | 0.544 / 0.694 / 1.038 | 0.411 / 0.623 / 0.952 | 737 / 1131 | +0.133 [+0.090, +0.182] | +0.071 [+0.030, +0.112] |
| M1 | midpoint_matching | 185,297 | 0.545 / 0.699 / 1.044 | 0.411 / 0.615 / 0.949 | 801 / 819 | +0.134 [+0.091, +0.182] | +0.084 [+0.050, +0.118] |
| M1 | common_same_identity | 183,935 | 0.544 / 0.693 / 1.037 | 0.410 / 0.612 / 0.945 | 690 / 753 |  |  |
| M1 | midpoint_only_or_changed | 1,362 | 0.794 / 1.309 / 1.619 | 0.656 / 0.998 / 1.330 | 111 / 66 |  |  |
| M1 | hsg_only_or_changed | 507 | 0.310 / 0.993 / 1.851 | 2.128 / 2.331 / 3.144 | 47 / 378 |  |  |
| M6 | released_hsg_matching | 134,382 | 0.701 / 0.903 / 1.350 | 0.705 / 0.960 / 1.367 | 3032 / 3512 | -0.004 [-0.063, +0.056] | -0.057 [-0.124, +0.017] |
| M6 | midpoint_matching | 134,465 | 0.701 / 0.907 / 1.352 | 0.705 / 0.940 / 1.360 | 2996 / 3237 | -0.004 [-0.064, +0.052] | -0.033 [-0.101, +0.047] |
| M6 | common_same_identity | 134,084 | 0.699 / 0.899 / 1.345 | 0.704 / 0.939 / 1.360 | 2838 / 3219 |  |  |
| M6 | midpoint_only_or_changed | 381 | 1.895 / 2.538 / 3.704 | 1.067 / 1.250 / 1.673 | 158 / 18 |  |  |
| M6 | hsg_only_or_changed | 298 | 2.261 / 2.162 / 2.701 | 4.388 / 4.334 / 5.110 | 194 / 293 |  |  |
| Ko-PER | released_hsg_matching | 25,122 | 0.312 / 0.579 / 0.758 | 0.381 / 0.688 / 0.952 | 171 / 325 | -0.068 [-0.138, -0.007] | -0.110 [-0.204, -0.044] |
| Ko-PER | midpoint_matching | 25,122 | 0.312 / 0.579 / 0.758 | 0.381 / 0.688 / 0.952 | 171 / 325 | -0.068 [-0.138, -0.007] | -0.110 [-0.204, -0.044] |
| Ko-PER | common_same_identity | 25,122 | 0.312 / 0.579 / 0.758 | 0.381 / 0.688 / 0.952 | 171 / 325 |  |  |
| Ko-PER | midpoint_only_or_changed | 0 | | | | | |
| Ko-PER | hsg_only_or_changed | 0 | | | | | |

## Reproduction of the stored attachment with HSG coordinates

| Recording | Camera | Stored rows | Reproduced rows | Keys only stored / only reproduced | Max reference gap (m) | Tracks assigned HSG / midpoint / total | Identity changed | Only HSG | Only midpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | C08 | 107,198 | 107,198 | 0 / 0 | 1.23e-03 | 603 / 602 / 986 | 0 | 8 | 7 |
| M1 | C09 | 223,852 | 223,852 | 0 / 0 | 6.46e-04 | 721 / 725 / 1434 | 0 | 7 | 11 |
| M1 | C10 | 226,033 | 226,034 | 0 / 1 | 1.01e-03 | 700 / 707 / 1345 | 0 | 3 | 10 |
| M6 | C05 | 165,515 | 165,514 | 1 / 0 | 1.06e-03 | 890 / 876 / 1675 | 0 | 18 | 4 |
| M6 | C06 | 133,143 | 133,143 | 0 / 0 | 1.68e-03 | 568 / 543 / 919 | 0 | 28 | 3 |
| M6 | C07 | 126,681 | 126,681 | 0 / 0 | 2.13e-04 | 604 / 593 / 921 | 0 | 16 | 5 |
| Ko-PER | SK1 | 46,082 | 46,082 | 0 / 0 | 1.08e-04 | 262 / 261 / 434 | 0 | 1 | 0 |
| Ko-PER | SK4 | 37,721 | 37,721 | 0 / 0 | 7.93e-04 | 275 / 273 / 339 | 0 | 2 | 0 |

The released report stores the clock map rounded to 0.1 ms and 0.01 ms/min, which is also what the released tracking evaluation reads; reproduction therefore agrees up to that rounding (millimetre-level reference gaps and, where a frame sits exactly on the 0.12 s or 10 m gate, single-frame key differences).

Stop rule: if the sets, identities, and core differences are stable and the released clock diagnostics show no anomaly, stage B (clock re-estimation) is not run. This reports sensitivity of the matching to the anchor rule under the released clocks; it does not establish clock independence or neutrality of the whole scoring chain.
