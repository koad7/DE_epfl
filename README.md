# dE_epfl

#### Installation:

From the root of the stdcode folder `./stdcode/`:

1. Create a virtualenv `virtualenv venv`
2. Install the dependencies `pip install -r requirements.txt`

#### How it works:

1. Launch the virtualenv python and import the interview code `>>> import interview_code`
2. Launch the code with the aoi file path and years range as argument. e.g. `>>> interview_code.main('data/aoi.shp', [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010])`
3. After running the stacked rasters will be generated in a folder under `./stacked` the name of the folder will be indicated in the terminal. The monthly rain averages will be given in the terminal as well.
