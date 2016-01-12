# NCAA FBS Projections

**REIFF** - Regression Estimated Iterative Football Forecaster

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models

*2015 NCAA FBS Bowl Projections*

### Overview

Head-to-head points

```
Accuracy: 0.560975609756
Against Spread: 0.317073170732
```

Point differential

```
Accuracy: 0.634146341463
Against Spread: 0.512195121951
```

### Head-to-head points

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team, projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

```
Arizona, 0.5796, New Mexico, 0.4204
-4.70832746107
Utah, 0.5178, BYU, 0.4822
-1.15592238906
Appalachian State, 0.7103, Ohio, 0.2895
-13.3554004411
San Jose State, 0.55, Georgia State, 0.45
-1.61190897552
Louisiana Tech, 0.5065, Arkansas State, 0.4932
-0.425221271289
Western Kentucky, 0.6782, South Florida, 0.3218
-9.70518851225
Utah State, 0.6105, Akron, 0.3893
-5.43898401827
Toledo, 0.6502, Temple, 0.3498
-5.85408234028
Boise State, 0.5546, Northern Illinois, 0.4454
-3.46417015238
Bowling Green, 0.7567, Georgia Southern, 0.2433
-9.23205410155
Western Michigan, 0.5182, Middle Tennessee, 0.48
-1.18105545451
Cincinnati, 0.5188, San Diego State, 0.4812
-0.904666620114
Marshall, 0.6683, Connecticut, 0.3273
-8.77809630324
Washington State, 0.6059, Miami (Florida), 0.3941
-5.10600932899
Southern Mississippi, 0.6713, Washington, 0.3277
-12.8832489116
Indiana, 0.5696, Duke, 0.4304
-2.70289211059
Tulsa, 0.6677, Virginia Tech, 0.3323
-7.34450675929
UCLA, 0.5758, Nebraska, 0.4241
-4.75186681086
Navy, 0.8169, Pittsburgh, 0.1831
-11.4341089584
Central Michigan, 0.5831, Minnesota, 0.4169
-3.45498669754
California, 0.5788, Air Force, 0.4212
-3.85473195556
Baylor, 0.6038, North Carolina, 0.3962
-6.30183683627
Colorado State, 0.7471, Nevada, 0.2529
-8.25830224046
Texas Tech, 0.7309, LSU, 0.269
-12.1460670302
Memphis, 0.693, Auburn, 0.2943
-19.7671662595
Mississippi State, 0.5258, North Carolina State, 0.4742
-1.18370687576
Texas A&M, 0.5066, Louisville, 0.4934
-0.27183639462
USC, 0.6636, Wisconsin, 0.3364
-8.10956161707
Houston, 0.6387, Florida State, 0.3613
-8.20379030129
Oklahoma, 0.6773, Clemson, 0.3227
-8.64367001802
Alabama, 0.6037, Michigan State, 0.3963
-2.84366964145
Tennessee, 0.7224, Northwestern, 0.2776
-14.0202172743
Notre Dame, 0.5476, Ohio State, 0.4524
-1.98945414198
Michigan, 0.5983, Florida, 0.4016
-5.83115662033
Stanford, 0.654, Iowa, 0.346
-6.57790304208
Oklahoma State, 0.5293, Mississippi, 0.4707
-1.77293666195
Penn State, 0.5091, Georgia, 0.4907
-0.360030329768
Arkansas, 0.5623, Kansas State, 0.4377
-3.35681345504
TCU, 0.57, Oregon, 0.43
-3.98094296189
West Virginia, 0.543, Arizona State, 0.457
-2.24814843635
Clemson, 0.5877, Alabama, 0.4123
-3.56158002366
Accuracy: 0.560975609756
Against Spread: 0.317073170732
```

### Point differential

"Defense wins championships" and clearly there is room for improvement by accounting for a defense's contribution to winning. Projecting margin of victory instead of total point production yields a 5% improvement in win-loss accuracy (63.41% > 56.1%). This model is also a Monte Carlo model simulated with KDE and Lasso regression, but projects point margin instead of total points. Spreads are the median margin of victory. A tie breaking strategy is currently not implemented; ties contribute toward each team's win probability, which is why probabilities do not equal 1.

```
New Mexico, 0.6207, Arizona, 0.5818
-1.50076420858
BYU, 0.5765, Utah, 0.4908
-13.694616364
Appalachian State, 0.6263, Ohio, 0.5068
-5.70054296494
Georgia State, 0.6227, San Jose State, 0.6095
-2.0641339626
Louisiana Tech, 0.6135, Arkansas State, 0.525
-5.89564595176
Western Kentucky, 0.6597, South Florida, 0.3754
-11.1402723041
Akron, 0.608, Utah State, 0.6
-2.28628833357
Toledo, 0.5798, Temple, 0.4794
-13.0647490854
Boise State, 0.5936, Northern Illinois, 0.5041
-12.9984286942
Bowling Green, 0.6939, Georgia Southern, 0.3729
-8.870420153
Middle Tennessee, 0.6593, Western Michigan, 0.5444
-2.61620899932
San Diego State, 0.6584, Cincinnati, 0.4737
-6.18491066511
Marshall, 0.6926, Connecticut, 0.4839
-0.0
Miami (Florida), 0.6173, Washington State, 0.5665
-5.83007595297
Southern Mississippi, 0.6656, Washington, 0.4779
-7.69936127371
Duke, 0.7419, Indiana, 0.4808
-0.0
Virginia Tech, 0.7821, Tulsa, 0.401
-0.0
UCLA, 0.6535, Nebraska, 0.4354
-5.54369386922
Navy, 0.6782, Pittsburgh, 0.4119
-6.86723703292
Central Michigan, 0.6439, Minnesota, 0.5206
-0.0
Air Force, 0.5851, California, 0.5312
-7.75348294082
Baylor, 0.5431, North Carolina, 0.4716
-18.6296368544
Colorado State, 0.7056, Nevada, 0.4563
-1.40535406857
LSU, 0.6164, Texas Tech, 0.6156
-0.0
Memphis, 0.8033, Auburn, 0.3361
-0.0
Mississippi State, 0.5898, North Carolina State, 0.4962
-9.19639083209
Louisville, 0.6055, Texas A&M, 0.598
-3.24225062753
Wisconsin, 0.583, USC, 0.5331
-9.54816175024
Houston, 0.6085, Florida State, 0.4413
-10.1300014409
Oklahoma, 0.5994, Clemson, 0.4178
-21.7693619435
Alabama, 0.5865, Michigan State, 0.4585
-13.0676772769
Tennessee, 0.5802, Northwestern, 0.4557
-12.6207704135
Ohio State, 0.7174, Notre Dame, 0.2838
-12.8248790971
Michigan, 0.6778, Florida, 0.4247
-7.72340743036
Stanford, 0.523, Iowa, 0.5175
-14.399476052
Mississippi, 0.6242, Oklahoma State, 0.4654
-8.30384406221
Penn State, 0.5477, Georgia, 0.5253
-11.0597488147
Arkansas, 0.7145, Kansas State, 0.564
-0.0
TCU, 0.656, Oregon, 0.4359
-7.44151994921
West Virginia, 0.7239, Arizona State, 0.404
-1.92454994014
Clemson, 0.5643, Alabama, 0.4513
-18.4814595865
Accuracy: 0.634146341463
Against Spread: 0.512195121951
```

## Footnotes

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2012 NFL first-round selection](https://en.wikipedia.org/wiki/Riley_Reiff) Riley Reiff. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind). Team data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/). Odds data is from [TeamRankings.com](https://www.teamrankings.com/college-football-bowls/schedule/).
