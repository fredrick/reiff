# NCAA FBS Projections

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models, 2015 NCAA FBS Bowl Projections

REIFF: Regression Estimated Iterative Football Forecaster

### Head-to-head points

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team, projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

```
Arizona, 0.5909, New Mexico, 0.409
-5.10205391685
Utah, 0.5197, BYU, 0.4803
-1.40495027788
Appalachian State, 0.7217, Ohio, 0.2782
-14.2950764182
San Jose State, 0.5488, Georgia State, 0.4512
-1.6743529751
Louisiana Tech, 0.5084, Arkansas State, 0.4911
-0.51048252427
Western Kentucky, 0.6801, South Florida, 0.3199
-9.5265664331
Utah State, 0.6102, Akron, 0.3897
-5.45903755493
Toledo, 0.6528, Temple, 0.3472
-5.8195441897
Boise State, 0.5457, Northern Illinois, 0.4543
-2.73396022092
Bowling Green, 0.7646, Georgia Southern, 0.2354
-9.37515176096
Western Michigan, 0.5184, Middle Tennessee, 0.4791
-1.48186066828
Cincinnati, 0.5154, San Diego State, 0.4846
-0.752026976995
Marshall, 0.6693, Connecticut, 0.327
-9.10419021138
Washington State, 0.6056, Miami (Florida), 0.3944
-5.0956074314
Southern Mississippi, 0.6808, Washington, 0.3185
-13.2736487419
Indiana, 0.5679, Duke, 0.4321
-2.90259401136
Tulsa, 0.6657, Virginia Tech, 0.3343
-7.63064991657
UCLA, 0.5699, Nebraska, 0.4298
-4.66428077901
Navy, 0.8104, Pittsburgh, 0.1896
-11.3283077483
Central Michigan, 0.5793, Minnesota, 0.4207
-3.11141166191
California, 0.5826, Air Force, 0.4174
-4.1467251157
Baylor, 0.5996, North Carolina, 0.4004
-5.7581097797
Colorado State, 0.7436, Nevada, 0.2564
-8.13862478508
Texas Tech, 0.7329, LSU, 0.2671
-12.1010285265
Memphis, 0.6904, Auburn, 0.2956
-19.4687018963
Mississippi State, 0.5346, North Carolina State, 0.4654
-1.38837137374
Texas A&M, 0.5033, Louisville, 0.4967
-0.183641139846
USC, 0.6621, Wisconsin, 0.3379
-7.93844873447
Houston, 0.6462, Florida State, 0.3538
-8.07908030747
Oklahoma, 0.678, Clemson, 0.322
-8.62034464974
Alabama, 0.6022, Michigan State, 0.3978
-2.91719301743
Tennessee, 0.7198, Northwestern, 0.2802
-13.1980887081
Notre Dame, 0.5582, Ohio State, 0.4418
-2.08100959318
Michigan, 0.5963, Florida, 0.4037
-5.82819607511
Stanford, 0.651, Iowa, 0.349
-6.69720118444
Oklahoma State, 0.5267, Mississippi, 0.4733
-1.547643611
Penn State, 0.504, Georgia, 0.4959
-0.132822497615
Arkansas, 0.5593, Kansas State, 0.4406
-3.28973106802
TCU, 0.5796, Oregon, 0.4204
-4.42243191613
West Virginia, 0.5456, Arizona State, 0.4544
-2.48718683329
Clemson, 0.5875, Alabama, 0.4125
-3.48766653026
Accuracy: 0.575
Against Spread: 0.325
```

### Margin of victory

"Defense wins championships" and clearly there is room for improvement by accounting for a defense's contribution to winning. Projecting margin of victory instead of total point production yields a 5% improvement in win-loss accuracy (62.5% > 57.5%). This model is also a Monte Carlo model simulated with KDE and Lasso regression, but projects point margin instead of total points. Spreads are the median margin of victory. A tie breaking strategy is currently not implemented; ties contribute toward each team's win probability, which is why probabilities do not equal 1.

```
New Mexico, 0.6265, Arizona, 0.5816
-0.588294640576
BYU, 0.5769, Utah, 0.4897
-14.1334519912
Appalachian State, 0.6258, Ohio, 0.4976
-5.83152046968
Georgia State, 0.6231, San Jose State, 0.6115
-2.14782406204
Louisiana Tech, 0.6041, Arkansas State, 0.5345
-5.97026067881
Western Kentucky, 0.6507, South Florida, 0.3827
-11.4605191773
Akron, 0.6235, Utah State, 0.5926
-1.41943810713
Toledo, 0.5681, Temple, 0.4819
-13.2062021237
Boise State, 0.592, Northern Illinois, 0.5085
-12.1691373063
Bowling Green, 0.6899, Georgia Southern, 0.3695
-8.90334493945
Middle Tennessee, 0.6554, Western Michigan, 0.5393
-2.68587276925
San Diego State, 0.6546, Cincinnati, 0.4703
-6.53048026101
Marshall, 0.6919, Connecticut, 0.4757
-0.0
Miami (Florida), 0.6042, Washington State, 0.5902
-6.05261638372
Southern Mississippi, 0.6656, Washington, 0.4861
-6.96373272343
Duke, 0.7409, Indiana, 0.4838
-0.0
Virginia Tech, 0.7826, Tulsa, 0.3998
-0.0
UCLA, 0.6426, Nebraska, 0.4505
-6.07256201576
Navy, 0.6846, Pittsburgh, 0.3998
-6.71769550675
Central Michigan, 0.6461, Minnesota, 0.5281
-0.0
Air Force, 0.5673, California, 0.5425
-8.23825137676
Baylor, 0.5417, North Carolina, 0.4712
-18.6791963374
Colorado State, 0.703, Nevada, 0.4535
-1.77864063243
LSU, 0.6176, Texas Tech, 0.614
-0.0
Memphis, 0.8081, Auburn, 0.342
-0.0
Mississippi State, 0.5829, North Carolina State, 0.5029
-9.77331210509
Louisville, 0.598, Texas A&M, 0.5978
-3.5793312059
Wisconsin, 0.58, USC, 0.5347
-10.1137618692
Houston, 0.6118, Florida State, 0.4368
-9.56463401231
Oklahoma, 0.6032, Clemson, 0.4167
-21.4061249965
Alabama, 0.5869, Michigan State, 0.4603
-12.2387002477
Tennessee, 0.5875, Northwestern, 0.4518
-12.1740785688
Ohio State, 0.7102, Notre Dame, 0.2908
-12.8648634989
Michigan, 0.6853, Florida, 0.4162
-7.48171649485
Stanford, 0.5241, Iowa, 0.518
-14.6536392766
Mississippi, 0.6221, Oklahoma State, 0.4704
-7.88578272791
Penn State, 0.5383, Georgia, 0.534
-11.4068981357
Arkansas, 0.711, Kansas State, 0.5596
-0.0
TCU, 0.6589, Oregon, 0.4393
-7.51333200008
West Virginia, 0.7208, Arizona State, 0.4135
-1.96104650557
Clemson, 0.5591, Alabama, 0.4605
-18.5566055762
Accuracy: 0.65
Against Spread: 0.525
```

## About

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2012 NFL first-round selection](https://en.wikipedia.org/wiki/Riley_Reiff) Riley Reiff. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind). Team data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/). Odds data is from [TeamRankings.com](https://www.teamrankings.com/college-football-bowls/schedule/).
