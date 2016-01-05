# NCAA FBS Projections

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models, 2015 NCAA FBS Bowl Projections

REIFF: Regression Estimated Iterative Football Forecaster

### Head-to-head points

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team, projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

```
Arizona, 0.5823, New Mexico, 0.4176
-4.86186638608
Utah, 0.523, BYU, 0.477
-1.58749408247
Appalachian State, 0.7147, Ohio, 0.2849
-13.5355076196
San Jose State, 0.5462, Georgia State, 0.4538
-1.62979224236
Louisiana Tech, 0.5012, Arkansas State, 0.4981
-0.0866116736266
Western Kentucky, 0.6835, South Florida, 0.3165
-9.80202755454
Utah State, 0.6227, Akron, 0.3773
-5.85172180056
Toledo, 0.6616, Temple, 0.3384
-6.06600468605
Boise State, 0.5414, Northern Illinois, 0.4586
-2.7344524841
Bowling Green, 0.762, Georgia Southern, 0.238
-9.57429276185
Western Michigan, 0.5038, Middle Tennessee, 0.4948
-0.249349472353
Cincinnati, 0.5236, San Diego State, 0.4764
-1.18289807011
Marshall, 0.6694, Connecticut, 0.3267
-9.12627145878
Washington State, 0.6164, Miami (Florida), 0.3836
-5.68117841343
Southern Mississippi, 0.6701, Washington, 0.3292
-12.3517760969
Indiana, 0.5789, Duke, 0.4211
-3.04364323825
Tulsa, 0.6643, Virginia Tech, 0.3357
-7.40762829944
UCLA, 0.5773, Nebraska, 0.4226
-5.06155966604
Navy, 0.8108, Pittsburgh, 0.1892
-11.4753730899
Central Michigan, 0.59, Minnesota, 0.41
-3.44040342851
California, 0.5788, Air Force, 0.4212
-3.79863214586
Baylor, 0.5995, North Carolina, 0.4005
-5.86273101586
Colorado State, 0.7499, Nevada, 0.2501
-8.16769451841
Texas Tech, 0.7344, LSU, 0.2655
-12.5012066989
Memphis, 0.6983, Auburn, 0.2875
-19.6063184886
Mississippi State, 0.5367, North Carolina State, 0.4633
-1.50339539315
Texas A&M, 0.5026, Louisville, 0.4974
-0.139771834111
USC, 0.6573, Wisconsin, 0.3427
-8.11397037589
Houston, 0.6483, Florida State, 0.3517
-8.38927160905
Oklahoma, 0.6662, Clemson, 0.3338
-8.53199774456
Alabama, 0.6032, Michigan State, 0.3968
-2.8477893252
Tennessee, 0.7169, Northwestern, 0.2831
-13.3438302496
Notre Dame, 0.5571, Ohio State, 0.4429
-1.9933428002
Michigan, 0.5843, Florida, 0.4157
-4.85660954061
Stanford, 0.6459, Iowa, 0.3541
-6.32887441036
Oklahoma State, 0.5254, Mississippi, 0.4746
-1.30181300872
Penn State, 0.5058, Georgia, 0.4942
-0.273679317202
Arkansas, 0.5732, Kansas State, 0.4264
-3.91929914909
TCU, 0.5648, Oregon, 0.4352
-3.90728994998
West Virginia, 0.5524, Arizona State, 0.4476
-2.8163903505
Clemson, 0.5775, Alabama, 0.4225
-3.2380219847
Accuracy: 0.575
Against Spread: 0.428571428571
```

### Margin of victory

"Defense wins championships" and clearly there is room for improvement by accounting for a defense's contribution to winning. Projecting margin of victory instead of total point production yields a 5% improvement in win-loss accuracy (62.5% > 57.5%). This model is also a Monte Carlo model simulated with KDE and Lasso regression, but projects point margin instead of total points. Spreads are the median margin of victory. A tie breaking strategy is currently not implemented; ties contribute toward each team's win probability, which is why probabilities do not equal 1.

```
New Mexico, 0.623, Arizona, 0.5804
-1.39126808067
BYU, 0.5809, Utah, 0.4963
-13.4243524671
Appalachian State, 0.6201, Ohio, 0.5052
-5.90375247552
Georgia State, 0.6312, San Jose State, 0.6055
-2.1093893813
Louisiana Tech, 0.6093, Arkansas State, 0.5216
-6.01892789647
Western Kentucky, 0.6591, South Florida, 0.3747
-11.0931620684
Akron, 0.6215, Utah State, 0.5975
-1.03153330801
Toledo, 0.5838, Temple, 0.4665
-13.042647329
Boise State, 0.5975, Northern Illinois, 0.4999
-12.3885188924
Bowling Green, 0.6989, Georgia Southern, 0.3678
-9.03385840368
Middle Tennessee, 0.6541, Western Michigan, 0.5462
-2.5787997742
San Diego State, 0.6533, Cincinnati, 0.4811
-6.37171313122
Marshall, 0.687, Connecticut, 0.4891
-0.0
Miami (Florida), 0.597, Washington State, 0.5814
-6.51520110779
Southern Mississippi, 0.666, Washington, 0.4855
-6.68187547811
Duke, 0.7398, Indiana, 0.4822
-0.0
Virginia Tech, 0.7821, Tulsa, 0.4046
-0.0
UCLA, 0.6453, Nebraska, 0.4561
-5.88497517991
Navy, 0.6826, Pittsburgh, 0.4028
-6.69121476531
Central Michigan, 0.6409, Minnesota, 0.5305
-0.0
Air Force, 0.5693, California, 0.5457
-8.59593366554
Baylor, 0.5423, North Carolina, 0.4742
-18.6048591479
Colorado State, 0.7024, Nevada, 0.4515
-1.77278753898
Texas Tech, 0.615, LSU, 0.6108
-3.67577884735
Memphis, 0.8023, Auburn, 0.3412
-0.0
Mississippi State, 0.5867, North Carolina State, 0.4984
-9.32713468427
Louisville, 0.6054, Texas A&M, 0.5957
-3.37133773162
Wisconsin, 0.5802, USC, 0.538
-9.80635126046
Houston, 0.6109, Florida State, 0.4416
-9.78141431346
Oklahoma, 0.6002, Clemson, 0.4182
-21.8374289328
Alabama, 0.5872, Michigan State, 0.457
-12.2795117922
Tennessee, 0.5879, Northwestern, 0.4501
-11.9314764673
Ohio State, 0.7106, Notre Dame, 0.2905
-12.9924893912
Michigan, 0.6774, Florida, 0.4241
-7.39283627921
Stanford, 0.5291, Iowa, 0.5142
-14.5852325569
Mississippi, 0.6211, Oklahoma State, 0.4699
-8.70109454315
Georgia, 0.5393, Penn State, 0.5344
-9.49078626883
Arkansas, 0.7136, Kansas State, 0.5663
-0.0
TCU, 0.6634, Oregon, 0.4291
-7.30813555184
West Virginia, 0.7211, Arizona State, 0.4167
-1.63608889778
Clemson, 0.5474, Alabama, 0.4698
-18.6780617842
Accuracy: 0.65
Against Spread: 0.785714285714
```

## About

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. While not incorporated into these models, for bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/). Jupyter notebook has a working name in honor of [2012 NFL first-round selection](https://en.wikipedia.org/wiki/Riley_Reiff) Riley Reiff. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind). Team data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/). Odds data is from [TeamRankings.com](https://www.teamrankings.com/college-football-bowls/schedule/).
