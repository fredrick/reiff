# NCAA FBS Projections

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models, 2015 NCAA FBS Bowl Projections

### Head-to-head points

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team, projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

```
Arizona, 0.5854, New Mexico, 0.4145
-4.96372630803
Utah, 0.5085, BYU, 0.4915
-0.540930470751
Appalachian State, 0.7204, Ohio, 0.2794
-13.9269237065
San Jose State, 0.5471, Georgia State, 0.4529
-1.52651728645
Louisiana Tech, 0.5041, Arkansas State, 0.4956
-0.236102610183
Western Kentucky, 0.6861, South Florida, 0.3139
-10.1614554841
Utah State, 0.6118, Akron, 0.3881
-5.32199234559
Toledo, 0.6571, Temple, 0.3429
-6.08107947177
Boise State, 0.5455, Northern Illinois, 0.4545
-2.78007975954
Bowling Green, 0.7507, Georgia Southern, 0.2493
-9.29093635803
Western Michigan, 0.5201, Middle Tennessee, 0.4778
-1.39638920461
Cincinnati, 0.5247, San Diego State, 0.4753
-1.18917690475
Marshall, 0.6796, Connecticut, 0.3174
-9.5114768113
Washington State, 0.6055, Miami (Florida), 0.3945
-4.93852063194
Southern Mississippi, 0.6785, Washington, 0.3203
-13.3397461731
Indiana, 0.5616, Duke, 0.4384
-2.51747353523
Tulsa, 0.6638, Virginia Tech, 0.3362
-7.07771753203
UCLA, 0.5696, Nebraska, 0.4302
-4.900077284
Navy, 0.8147, Pittsburgh, 0.1853
-11.7242485566
Central Michigan, 0.5757, Minnesota, 0.4243
-3.01682646543
California, 0.5775, Air Force, 0.4225
-3.7110515643
Baylor, 0.6107, North Carolina, 0.3893
-6.38316915084
Colorado State, 0.7481, Nevada, 0.2519
-8.29363663816
Texas Tech, 0.7373, LSU, 0.2627
-12.2364404356
Memphis, 0.6928, Auburn, 0.2943
-19.5171189189
Mississippi State, 0.5353, North Carolina State, 0.4647
-1.51244579445
Texas A&M, 0.5061, Louisville, 0.4939
-0.285580171838
USC, 0.6593, Wisconsin, 0.3407
-8.10838145345
Houston, 0.6448, Florida State, 0.3552
-8.02688380023
Oklahoma, 0.6785, Clemson, 0.3215
-8.76708467752
Alabama, 0.6115, Michigan State, 0.3885
-3.17545841716
Tennessee, 0.7236, Northwestern, 0.2764
-13.43471006
Notre Dame, 0.5437, Ohio State, 0.4563
-1.57078532789
Michigan, 0.5951, Florida, 0.4048
-5.43259570255
Stanford, 0.6521, Iowa, 0.3479
-6.46355199018
Oklahoma State, 0.5312, Mississippi, 0.4688
-1.74843496377
Penn State, 0.5056, Georgia, 0.4944
-0.21805949765
Arkansas, 0.5707, Kansas State, 0.4293
-3.57934377589
TCU, 0.5637, Oregon, 0.4363
-3.77387156975
West Virginia, 0.5516, Arizona State, 0.4484
-2.60194009706
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

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2015 Jim Thorpe](http://www.blackheartgoldpants.com/2015/12/10/9890276/desmond-king-wins-jim-thorpe-award-for-best-defensive-back) and consensus All-American Desmond King. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind), just don't bet with it yet! Data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/).
