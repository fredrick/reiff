# NCAA FBS Projections

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models, 2015 NCAA FBS Bowl Projections

REIFF: Regression Estimated Iterative Football Forecaster

### Head-to-head points

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team, projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

```
Arizona, 0.5913, New Mexico, 0.4087
-5.30054822593
Utah, 0.5187, BYU, 0.4812
-1.13385798144
Appalachian State, 0.7255, Ohio, 0.2741
-14.0479956441
San Jose State, 0.5428, Georgia State, 0.4572
-1.54302027563
Louisiana Tech, 0.5072, Arkansas State, 0.4923
-0.522121422808
Western Kentucky, 0.668, South Florida, 0.332
-9.04857840087
Utah State, 0.6172, Akron, 0.3828
-5.77542781283
Toledo, 0.6518, Temple, 0.3482
-5.71886754818
Boise State, 0.5427, Northern Illinois, 0.4573
-2.69197662378
Bowling Green, 0.755, Georgia Southern, 0.245
-9.19202025275
Western Michigan, 0.518, Middle Tennessee, 0.4796
-1.30622358426
Cincinnati, 0.5266, San Diego State, 0.4734
-1.36392374216
Marshall, 0.6763, Connecticut, 0.3198
-9.34312457996
Washington State, 0.6044, Miami (Florida), 0.3956
-5.10000412776
Southern Mississippi, 0.6869, Washington, 0.3124
-13.8555623706
Indiana, 0.5631, Duke, 0.4369
-2.48402670949
Tulsa, 0.6706, Virginia Tech, 0.3294
-7.47671230182
UCLA, 0.5731, Nebraska, 0.4267
-4.83605289941
Navy, 0.8126, Pittsburgh, 0.1874
-11.3892684869
Central Michigan, 0.5904, Minnesota, 0.4096
-3.37192418699
California, 0.5778, Air Force, 0.4222
-3.66218495633
Baylor, 0.6026, North Carolina, 0.3974
-6.26602111817
Colorado State, 0.7395, Nevada, 0.2605
-8.0288664086
Texas Tech, 0.7293, LSU, 0.2707
-12.0459942702
Memphis, 0.701, Auburn, 0.2865
-19.8909165865
Mississippi State, 0.5364, North Carolina State, 0.4636
-1.53967728883
Texas A&M, 0.5069, Louisville, 0.4931
-0.299962805945
USC, 0.6572, Wisconsin, 0.3428
-7.98026953207
Houston, 0.6401, Florida State, 0.3599
-7.66933276237
Oklahoma, 0.6646, Clemson, 0.3354
-8.30510328517
Alabama, 0.6141, Michigan State, 0.3859
-3.2208931763
Tennessee, 0.7194, Northwestern, 0.2806
-13.5427890622
Notre Dame, 0.5495, Ohio State, 0.4505
-1.92367746792
Michigan, 0.5931, Florida, 0.4067
-5.5016893063
Stanford, 0.6471, Iowa, 0.3529
-6.40422482181
Oklahoma State, 0.5238, Mississippi, 0.4762
-1.3513101365
Penn State, 0.5055, Georgia, 0.4944
-0.233205235047
Arkansas, 0.5605, Kansas State, 0.4393
-3.36840895142
TCU, 0.5683, Oregon, 0.4317
-3.84253359836
West Virginia, 0.5465, Arizona State, 0.4535
-2.41831072781
Clemson, 0.5792, Alabama, 0.4208
-3.09168208473
Accuracy: 0.575
```

### Margin of victory

"Defense wins championships" and clearly there is room for improvement by accounting for a defense's contribution to winning. Projecting margin of victory instead of total point production yields a 5% improvement in win-loss accuracy (62.5% > 57.5%). This model is also a Monte Carlo model simulated with KDE and Lasso regression, but projects point margin instead of total points. Spreads are the median margin of victory. A tie breaking strategy is currently not implemented; ties contribute toward each team's win probability, which is why probabilities do not equal 1.

```
New Mexico, 0.6229, Arizona, 0.5739
-1.49506980399
BYU, 0.5854, Utah, 0.4898
-13.9871831884
Appalachian State, 0.6339, Ohio, 0.4952
-4.8399796237
Georgia State, 0.6289, San Jose State, 0.6015
-2.09376032142
Louisiana Tech, 0.6099, Arkansas State, 0.5266
-6.26895871662
Western Kentucky, 0.6637, South Florida, 0.3715
-11.3219354937
Akron, 0.6208, Utah State, 0.5975
-0.728243074279
Toledo, 0.5775, Temple, 0.4732
-12.9954364572
Boise State, 0.5958, Northern Illinois, 0.5011
-12.4446866863
Bowling Green, 0.6961, Georgia Southern, 0.3674
-8.66834636237
Middle Tennessee, 0.6522, Western Michigan, 0.547
-2.65792137988
San Diego State, 0.6461, Cincinnati, 0.4829
-6.51165179369
Marshall, 0.6936, Connecticut, 0.4812
-0.0
Miami (Florida), 0.5924, Washington State, 0.5896
-6.17827390876
Southern Mississippi, 0.6717, Washington, 0.4771
-7.29361226395
Duke, 0.7353, Indiana, 0.4916
-0.0
Virginia Tech, 0.7819, Tulsa, 0.4015
-0.0
UCLA, 0.6484, Nebraska, 0.4448
-5.82376451505
Navy, 0.6786, Pittsburgh, 0.3994
-6.71168097185
Central Michigan, 0.6292, Minnesota, 0.5363
-0.0
Air Force, 0.5772, California, 0.539
-7.92421322292
Baylor, 0.5447, North Carolina, 0.4718
-18.6199639661
Colorado State, 0.6967, Nevada, 0.4664
-1.61989715218
LSU, 0.6158, Texas Tech, 0.6109
-0.0
Memphis, 0.8009, Auburn, 0.338
-0.0
Mississippi State, 0.5761, North Carolina State, 0.5043
-9.9767969814
Texas A&M, 0.6021, Louisville, 0.601
-2.65439760233
Wisconsin, 0.578, USC, 0.5386
-10.0128025353
Houston, 0.6143, Florida State, 0.4358
-9.57017821593
Oklahoma, 0.6, Clemson, 0.4197
-21.7583632613
Alabama, 0.5893, Michigan State, 0.454
-12.3507695807
Tennessee, 0.5847, Northwestern, 0.4507
-12.3756736141
Ohio State, 0.7152, Notre Dame, 0.2858
-13.0200662342
Michigan, 0.6783, Florida, 0.4207
-7.67268350837
Stanford, 0.5304, Iowa, 0.5085
-14.665260081
Mississippi, 0.626, Oklahoma State, 0.4682
-8.96124081584
Penn State, 0.5425, Georgia, 0.5326
-11.0051052559
Arkansas, 0.7186, Kansas State, 0.5615
-0.0
TCU, 0.6614, Oregon, 0.4369
-7.07610980604
West Virginia, 0.7189, Arizona State, 0.4142
-1.86451770421
Clemson, 0.5615, Alabama, 0.4575
-18.4050807003
Accuracy: 0.625
```

## About

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2012 NFL first-round selection](https://en.wikipedia.org/wiki/Riley_Reiff) Riley Reiff. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind), just don't bet with it yet! Data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/).
