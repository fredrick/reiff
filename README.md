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
New Mexico, 0.6221, Arizona, 0.5842
-1.00206613716
BYU, 0.5819, Utah, 0.4852
-13.9993265179
Appalachian State, 0.6227, Ohio, 0.5089
-5.72324294684
Georgia State, 0.6236, San Jose State, 0.6066
-2.71634624629
Louisiana Tech, 0.6057, Arkansas State, 0.5274
-6.06628874465
Western Kentucky, 0.6465, South Florida, 0.3876
-11.5440001775
Akron, 0.6105, Utah State, 0.6082
-2.41851574183
Toledo, 0.5854, Temple, 0.4717
-12.7219937304
Boise State, 0.5888, Northern Illinois, 0.5064
-12.7746249088
Bowling Green, 0.6922, Georgia Southern, 0.3675
-8.98374894942
Middle Tennessee, 0.6567, Western Michigan, 0.5378
-2.82908890793
San Diego State, 0.6551, Cincinnati, 0.4697
-6.22956916209
Marshall, 0.6825, Connecticut, 0.4949
-0.0
Miami (Florida), 0.6051, Washington State, 0.5764
-6.16596619262
Southern Mississippi, 0.6696, Washington, 0.4765
-6.79633876495
Duke, 0.735, Indiana, 0.4835
-0.0
Virginia Tech, 0.7776, Tulsa, 0.4076
-0.0
UCLA, 0.6449, Nebraska, 0.4458
-6.03276651419
Navy, 0.6837, Pittsburgh, 0.3985
-6.80825046379
Central Michigan, 0.639, Minnesota, 0.5204
-0.0
Air Force, 0.5846, California, 0.5306
-8.01668947919
Baylor, 0.5426, North Carolina, 0.4736
-18.5168759723
Colorado State, 0.7039, Nevada, 0.4594
-1.62071048441
LSU, 0.6201, Texas Tech, 0.6141
-0.0
Memphis, 0.7994, Auburn, 0.3376
-0.0
Mississippi State, 0.5797, North Carolina State, 0.5034
-9.66312458632
Texas A&M, 0.5993, Louisville, 0.5967
-2.71676201037
Wisconsin, 0.5829, USC, 0.5258
-9.76435550827
Houston, 0.6093, Florida State, 0.4378
-9.98176638864
Oklahoma, 0.5987, Clemson, 0.4184
-22.0896704279
Alabama, 0.5989, Michigan State, 0.4448
-11.4617817134
Tennessee, 0.5778, Northwestern, 0.4584
-12.8060826872
Ohio State, 0.7031, Notre Dame, 0.298
-13.1891317375
Michigan, 0.6776, Florida, 0.4219
-7.72333271723
Stanford, 0.5296, Iowa, 0.511
-14.5173815608
Mississippi, 0.6349, Oklahoma State, 0.4517
-7.49519572186
Penn State, 0.5395, Georgia, 0.5287
-11.1724946657
Arkansas, 0.7069, Kansas State, 0.5641
-0.0
TCU, 0.6593, Oregon, 0.4339
-7.37910415047
West Virginia, 0.7233, Arizona State, 0.4133
-1.33746983329
Clemson, 0.5545, Alabama, 0.462
-18.2471791832
Accuracy: 0.625
Against Spread: 0.5
```

## About

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2012 NFL first-round selection](https://en.wikipedia.org/wiki/Riley_Reiff) Riley Reiff. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind). Team data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/). Odds data is from [TeamRankings.com](https://www.teamrankings.com/college-football-bowls/schedule/).
