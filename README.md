# NCAA FBS Projections

Quick, dirty, and very rough experiments with Python, projection models, machine learning, and NCAA FBS football. Models implemented with [Jupyter/IPython](http://jupyter.readthedocs.org/), [Pandas](http://pandas.pydata.org/), [NumPy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Models, 2015 NCAA FBS Bowl Projections

### Head-to-head offense only

What if only offense mattered? This model explores this what if scenario using a Monte Carlo projected point model simulated with [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) and [Lasso](http://statweb.stanford.edu/~tibs/lasso/simple.html) regression. Output is projected winning team (offense), projected spread is median head-to-head point margin after simulating 10,000 games, followed by odds of each team winning.

"Defense wins championships" and clearly there is room for improvement by modeling an opponent's offense against a team's defense. For bowl games especially, strength of schedule, and team based rankings (e.g. FPI, Elo) have been shown to be [fairly effective](http://fivethirtyeight.com/features/heres-how-our-college-football-playoff-predictions-work/).

```
Arizona, 0.5889, New Mexico, 0.411
-5.03234456701
Utah, 0.518, BYU, 0.482
-1.2208513588
Appalachian State, 0.7168, Ohio, 0.2831
-13.7828220229
San Jose State, 0.5509, Georgia State, 0.4491
-1.69516445077
Louisiana Tech, 0.5116, Arkansas State, 0.4879
-0.988603515798
Western Kentucky, 0.6749, South Florida, 0.3251
-9.21396138608
Utah State, 0.6218, Akron, 0.3782
-5.86002686933
Toledo, 0.6649, Temple, 0.3351
-6.2352883198
Boise State, 0.5465, Northern Illinois, 0.4535
-3.16863494508
Bowling Green, 0.758, Georgia Southern, 0.242
-9.34113406384
Western Michigan, 0.5213, Middle Tennessee, 0.4758
-1.7379816171
Cincinnati, 0.509, San Diego State, 0.491
-0.46692297147
Marshall, 0.6704, Connecticut, 0.3266
-9.18643132285
Washington State, 0.601, Miami (Florida), 0.399
-5.14984781656
Southern Mississippi, 0.6775, Washington, 0.3213
-13.2893061507
Indiana, 0.5711, Duke, 0.4289
-2.77713608392
Tulsa, 0.6584, Virginia Tech, 0.3416
-7.14903836646
UCLA, 0.5689, Nebraska, 0.431
-4.39322243934
Navy, 0.816, Pittsburgh, 0.184
-11.5190238722
Central Michigan, 0.5891, Minnesota, 0.4109
-3.54091859037
California, 0.5778, Air Force, 0.4222
-3.74238348462
Baylor, 0.6028, North Carolina, 0.3972
-6.28159327371
Colorado State, 0.7433, Nevada, 0.2567
-8.08881422402
Texas Tech, 0.7344, LSU, 0.2656
-12.1508899857
Memphis, 0.698, Auburn, 0.289
-19.8848112251
Mississippi State, 0.5372, North Carolina State, 0.4628
-1.69926863167
Texas A&M, 0.5019, Louisville, 0.4981
-0.0886080596184
USC, 0.6603, Wisconsin, 0.3397
-8.02487376272
Houston, 0.6416, Florida State, 0.3583
-7.92687457407
Oklahoma, 0.6696, Clemson, 0.3304
-8.09466903919
Alabama, 0.6167, Michigan State, 0.3833
-3.17098721413
Tennessee, 0.729, Northwestern, 0.271
-14.2957564732
Notre Dame, 0.5478, Ohio State, 0.4522
-1.83322611516
Michigan, 0.5889, Florida, 0.4108
-5.02723229034
Stanford, 0.6462, Iowa, 0.3538
-6.30258242038
Oklahoma State, 0.5382, Mississippi, 0.4618
-2.07802175627
Penn State, 0.5, Georgia, 0.4999
-0.00272858025543
Arkansas, 0.5637, Kansas State, 0.4361
-3.57872272507
TCU, 0.5749, Oregon, 0.4251
-4.17631179548
West Virginia, 0.5527, Arizona State, 0.4473
-2.75995666835
Accuracy: 0.583333333333 (2015-12-30)
```

## About

College football is notoriously difficult to project due to the [small sample of data](http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node16.html) to draw from. Jupyter notebook has a working name in honor of [2015 Jim Thorpe](http://www.blackheartgoldpants.com/2015/12/10/9890276/desmond-king-wins-jim-thorpe-award-for-best-defensive-back) and consensus All-American Desmond King. You can probably guess what team I root for.

## Author

This project has been a fun weekend learning experience by [Fredrick Galoso](https://twitter.com/wayoutmind), just don't bet with it yet! Data is from [/r/CFBAnalysis](https://www.reddit.com/r/CFBAnalysis/comments/3j1gjg/2015_data_sources/).
