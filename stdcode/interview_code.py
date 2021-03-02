from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob
import itertools
import logging
import gzip
import rasterio
import rasterio.mask
import fiona
from multiprocessing import cpu_count
from bs4 import BeautifulSoup
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import requests
import sys
import subprocess
import pkg_resources
from datetime import datetime as DT


logging.basicConfig(filename='app.log', filemode='w',
                    format='%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s', level=logging.ERROR)


def project_init():
    '''
    Init function for folders creation and determining useful number of threads to use
    Parameters
    ----------
    ''
    Returns
    -------
    DOWNLOADS_DIR: Folder where the CHIRPS files are downloaded in 
    MASKED_FILES_DIR: Folder where the masked files are saved in
    SATCKED_FILES_DIR: Folder where the stacked files are saved in
    THREADS: Number of thread that the multithreading functions will use
    '''
    # CREATE WORK DIRS
    DOWNLOADS_DIR_TIF = './.tmp/downloads/tif/'  # create download files folder
    if not os.path.exists(DOWNLOADS_DIR_TIF):
        os.makedirs(DOWNLOADS_DIR_TIF)
    DOWNLOADS_DIR_AOI = './data/'  # given/downloaded  aoi files folder
    if not os.path.exists(DOWNLOADS_DIR_AOI):
        os.makedirs(DOWNLOADS_DIR_AOI)
    MASKED_FILES_DIR = './.tmp/masked/'  # create masked files folder
    if not os.path.exists(MASKED_FILES_DIR):
        os.makedirs(MASKED_FILES_DIR)
    SATCKED_FILES_DIR = './stacked/'
    if not os.path.exists(SATCKED_FILES_DIR):  # created stacked files folder
        os.makedirs(SATCKED_FILES_DIR)
    # Current stacked files
    SATCKED_FILES_CURRENT_DIR = SATCKED_FILES_DIR +  \
        DT.now().isoformat().replace(':', '_').split('.')[0] + '/'
    os.makedirs(SATCKED_FILES_CURRENT_DIR)
    # Nbr of thread to use for multithreading
    if (cpu_count() > 2):
        THREADS = cpu_count()-2
    else:
        THREADS = 2  # At least 2 for multithreading
    return {'DOWNLOADS_DIR_AOI': DOWNLOADS_DIR_AOI,
            'DOWNLOADS_DIR_TIF': DOWNLOADS_DIR_TIF,
            'MASKED_FILES_DIR': MASKED_FILES_DIR,
            'SATCKED_FILES_DIR': SATCKED_FILES_DIR, 'THREADS': THREADS,
            'SATCKED_FILES_CURRENT_DIR': SATCKED_FILES_CURRENT_DIR}


# project init
myenv = project_init()
DOWNLOADS_DIR_AOI = myenv['DOWNLOADS_DIR_AOI']
DOWNLOADS_DIR_TIF = myenv['DOWNLOADS_DIR_TIF']
MASKED_FILES_DIR = myenv['MASKED_FILES_DIR']
SATCKED_FILES_DIR = myenv['SATCKED_FILES_DIR']
SATCKED_FILES_CURRENT_DIR = myenv['SATCKED_FILES_CURRENT_DIR']
THREADS = myenv['THREADS']

# For the current files


def files_url_list(url, files, year):
    '''
    Build the files url list from the CHIRPS website. Use beautiful soup to extract urls from thewebsite html. 
    Parameters
    ----------
    url: url of the .tiff to download
    files: object to store the files url in
    year: year of selection
    Returns
    -------
    no return 
    '''
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    for node in soup.find_all('a'):
        try:
            if(node.get('href').endswith('tif') | node.get('href').endswith('gz')):  # select .tif or .gz files
                # selection of the year in the url
                if(node.get('href').split('.')[-4] == str(year) or node.get('href').split('.')[-5] == str(year)):
                    files.append(url + '/' + node.get('href'))
        except Exception as e:
            logging.exception(
                "files_url_list: Exception caught during processing")


def concurrent_files_url_list(baseUrl, years):
    '''
    Concurrent Donwload of .tiff file urls in a list
    Parameters
    ----------
    baseUrl: url of the page to download the files from
    years: list of year(s) of selection
    Returns
    -------
    files: list or urls to download
    '''
    files = {}
    # Concurent downloading of the data
    append_data = []
    result = []
    # Concurences
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for year in years:
            files[str(year)] = []
            # In case using daily rainfall data in the daily folder
            if(baseUrl.split('_')[-1].split('/')[0] == 'daily'):
                try:
                    executor.submit(files_url_list, baseUrl +
                                    str(year), files[str(year)], year)
                except Exception as e:
                    logging.exception(
                        "download_file_links: Exception caught during processing")
            # In case using daily rainfall data in the monthly folder
            elif(baseUrl.split('_')[-1].split('/')[0] == 'monthly'):
                try:
                    executor.submit(files_url_list, baseUrl,
                                    files[str(year)], year)
                except Exception as e:
                    logging.exception(
                        "download_file_links: Exception caught during processing")

    return files


def download_file(url, session):
    '''
    Download the .tif or .gz files and uncompress the .gz file in memory
    Parameters
    ----------
    url: url of the file to download
    Returns
    -------
    no return. Downloads files into DOWNLOADS_DIR_TIF
    '''
    if (url.endswith('gz')):
        outFilePath = DOWNLOADS_DIR_TIF+url.split('/')[-1][:-3]
    else:
        outFilePath = DOWNLOADS_DIR_TIF+url.split('/')[-1]
    response = session.get(url)
    with open(outFilePath, 'wb') as outfile:
        if (url.endswith('gz')):
            outfile.write(gzip.decompress(response.content))
        elif (url.endswith('tif')):
            outfile.write(response.content)
        else:
            pass


def concurrent_file_downloader(files):
    '''
    Concurent downloading and extraction of the data
    Parameters
    ----------
    files: list of url to download
    Returns
    -------
    no return
    '''
    session = requests.session()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    append_data = []
    result = []
    # Concurences
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for year in files:
            for url in files[year]:
                try:
                    executor.submit(download_file, url, session)
                except Exception as e:
                    logging.exception(
                        "concurrent_file_downloader: Exception caught during processing")


def aoi_shapefile_reader(aoishapefile):
    '''
    Download the shapefile of the area of interest (aoi)
    Parameters
    ----------
    aoishapefile: path to the shapefile to download
    Returns
    -------
    shapes: file containing the coordinates of the aoi's polygon
    '''
    # Read the AOI's shapefile
    with fiona.open(aoishapefile, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    return shapes


def masking(file, shapes, years):
    '''Masking of .tif files by the provided shapefile
    Parameters
    ----------
    file: .tif file to be masked
    shapes: Coordinate of the aoi polygon
    Returns
    -------
    export files into MASKED_FILES_DIR directory
    '''
    if (int(file.split('/')[-1].split('.')[-4]) in years):  # select the right year
        if file[-4:] == '.tif':
            with rasterio.open(DOWNLOADS_DIR_TIF+file) as src:
                out_image, out_transform = rasterio.mask.mask(
                    src, shapes, crop=True)
                out_meta = src.meta
            # use the updated spatial transform and raster height and width to write the masked raster to a new file.
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            with rasterio.open(MASKED_FILES_DIR+file[:-4]+".masked.tif", "w", **out_meta) as dest:
                dest.write(out_image)


def concurrent_masking(shapes, years):
    '''
    Launch the concurent masking of the list of .tiff files by the aio provided
    Parameters
    ----------
    shapes: Coordinate of the aoi polygon
    Returns
    -------
    '''
    append_data = []
    result = []
    # Concurent masking
    with ThreadPoolExecutor(max_workers=10) as executor:
        for file in os.listdir(DOWNLOADS_DIR_TIF):
            try:
                executor.submit(masking, file, shapes, years)
            except Exception as e:
                logging.exception(
                    "concurrent_file_downloader: Exception caught during processing")


def calculate_rainy_days(baseUrl, years):
    '''
    Calculate the number of rainy dates in month over a year
    Parameters
    ----------
    years: List of years selected
    Returns
    -------
    OrderedDict of month - rainy days
    '''
    MONTHS_DICT = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                   6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    data_array = []
    rainy_days = {}
    Mat = pd.DataFrame()
    table = pd.DataFrame(index=np.arange(0, 1))
    # Read rain data into dataframe
    for file in os.listdir(MASKED_FILES_DIR):
        if file[-4:] == '.tif':
            if (int(file[12:16]) in years):  # file of selected years only
                dataset = rasterio.open(MASKED_FILES_DIR+file)
                widht = dataset.profile.get('width')
                height = dataset.profile.get('height')
                data_array = dataset.read(1)  # read one band
                data_array_sparse = sparse.coo_matrix(
                    data_array, shape=(height, widht))
                if(baseUrl.split('_')[-1].split('/')[0] == 'daily'):
                    data = file[12:-11]
                elif(baseUrl.split('_')[-1].split('/')[0] == 'monthly'):
                    data = file[12:-14]
                Mat[data] = data_array_sparse.toarray().tolist()

    # Calculate the precipitaion per day dataframe
    # sum the array in each dataframe cells
    raindatadf = pd.DataFrame(Mat.applymap(
        lambda x: sum([t for t in x])).sum()).T
    for year in years:
        number_of_days = {}
        for i in range(1, 13):  # for 12 months
            number_of_days[MONTHS_DICT[i]] = raindatadf[raindatadf.columns[raindatadf.columns.str.slice(
                0, 7).str.endswith(f'{year}.{i:02}')]].gt(0.0).sum(axis=1)[0]
        rainy_days[year] = number_of_days
    return rainy_days


def order_masked_files_per_month():
    '''
    Order masked .tiff files from MASKED_FILES_DIR  per month.
    Parameters
    ----------
    years: List of years selected
    Returns
    -------
    rasterfiles: dictionnary of list of masked .tiff files ordered by month
    '''
    rasterfiles = {}
    def key(x): return x[17:19]  # Month
    masked_raster = os.listdir(MASKED_FILES_DIR)
    masked_raster = sorted(masked_raster, key=key)
    myfileslist = []

    for key, group in itertools.groupby(masked_raster, key):
        myfileslist.append(list(group))
    # Month selection as key
    for l in myfileslist:
        rasterfiles[str(l[0][17:19])] = l
    return rasterfiles


def stack_rasters(filelist, month, years):
    '''
    Stack list of .tif files by month
    Parameters
    ----------
    filelist: List of masked tiff images
    month: month selected in number ('01', '02'...'12')
    Returns
    -------
    no return. Copy the stacked .tiff images in SATCKED_FILES_DIR
    '''
    months_dict = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
                   '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
    with rasterio.open(MASKED_FILES_DIR+filelist[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(filelist))

    # Read each layer and write it to stack
    yearslist = '_'.join(map(str, [years[-1], years[0]]))
    with rasterio.open(f'{SATCKED_FILES_CURRENT_DIR}stacked_{yearslist}.{months_dict[month]}.tif', 'w', **meta) as dst:
        for id, layer in enumerate(filelist, start=1):
            with rasterio.open(MASKED_FILES_DIR+layer) as src1:
                dst.write_band(id, src1.read(1))


def concurrent_stack_rasters(rasterfiles, years):
    '''
    Launch the concurent stacking of the tiff images
    Parameters
    ----------
    No parameters
    Returns
    -------
    No return. Copy the stacked .tiff images in SATCKED_FILES_DIR
    '''
    appenddata = []
    result = []
    # Concurent masking
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for month in rasterfiles:
            try:
                # int(month[:3]) # trick to remove .DS_STORE file
                filelist = rasterfiles[month]
                executor.submit(stack_rasters, filelist, month, years)
            except Exception as e:
                pass


def delete_all_downloaded_files(filedir):
    '''
    Delete all files in given folder to free space.
    folderpath: folder path
    Parameters
    ----------
    folderpath : Path to the directory to erase files from
    Returns
    -------
    '''
    files = glob.glob(filedir + '*')
    for f in files:
        os.remove(f)


def main(aoifilepath, years):
    '''
    Generate the average rain days per selected years and the stacked files in the './stacked' folder.
    Parameters
    ----------
    aoifilepath : Path to the aoi file. e.g. 'data/aoi.shp',
    years: list of years e.g [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]
    Returns
    -------
    a dict of rainy days monthly means over the years period given.
    '''
    # Selection of year(s) of interest
    availableyears = [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002,
                      2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990, 1989, 1988, 1987, 1986, 1985, 1984, 1983, 1982, 1981]

    BASE_URL = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p25/'
    # check if selected years is in the available years
    if (all(item in availableyears for item in years)):
        files = {}
        # get all files ulrs from the CHIRPS dataset base_url
        files = concurrent_files_url_list(BASE_URL, years)
        print('1/6- Collecting .tif images links from CHIRPS webpage')
        # # launch concurent dowload of all the .tif files selected
        print('2/6- Dowloading the .tif files')
        concurrent_file_downloader(files)
        # dowload aoi file
        print('3/6- Loading aoi shape file')
        aoishapes = aoi_shapefile_reader(aoifilepath)

        # clipping or maksing.  The files are stored in MASKED_FILES_DIR
        print('4/6- Masking the .tif files with the aoi polygon')
        concurrent_masking(aoishapes, years)

        # Calculate number of raining days
        print('5/6- Calculate the rain days averages')
        raindata = calculate_rainy_days(BASE_URL, years)

        # order the masked files by year and month
        rasterfiles = order_masked_files_per_month()

        # Generate the stacked files in SATCKED_FILES_DIR
        print(
            f'6/6- Generating the stacked files in {SATCKED_FILES_CURRENT_DIR}')
        concurrent_stack_rasters(rasterfiles, years)
        print(
            f'Your stacked files are available here {SATCKED_FILES_CURRENT_DIR} ')
        print('Monthly rainy days average:')
        print(pd.DataFrame(raindata).mean(axis=1).round(decimals=0).astype(int))
        delete_all_downloaded_files(DOWNLOADS_DIR_TIF)
    else:
        print('The year(s) you have chosen are not part of the available data')

    print('... deleting the downloaded .tif files')

    return pd.DataFrame(raindata).mean(axis=1).round(decimals=0).astype(int).to_dict()
