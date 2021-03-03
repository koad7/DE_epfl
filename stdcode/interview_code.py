from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob
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
        i = 0
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


def stack_rasters(years):
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
    Mat = pd.DataFrame()
    ras_meta = {}  # Profile
    # Read rain data into dataframe
    # Read rain data into dataframe
    for file in os.listdir(MASKED_FILES_DIR):
        if file[-4:] == '.tif':
            if (int(file[12:16]) in years):  # file of selected years only
                # read the masked/clipped .tiff
                dataset = rasterio.open(MASKED_FILES_DIR+file)
                ras_meta = dataset.profile
                widht = dataset.profile.get('width')  # get raster dimensions
                height = dataset.profile.get('height')
                data_array = dataset.read(1)  # read one band
                data_array_sparse = sparse.coo_matrix(  # use scipy cordinate matrix/sparse matrix for better performance
                    data_array, shape=(height, widht))
                dates = file[12:-11]  # get dates
                # build Dataframe with date as colname and rain values
                Mat[dates] = data_array_sparse.toarray().tolist()

    Mat2 = pd.DataFrame(Mat.applymap(lambda x: [1 if l > 0 else 0 for l in x]))

    number_of_days = {}
    for i in range(1, len(MONTHS_DICT)+1):  # for 12 months
        number_of_days[MONTHS_DICT[i]] = Mat2[Mat2.columns[Mat2.columns.str.slice(
            0, 7).str.endswith(f'{i:02}')]].applymap(np.array).sum(axis=1)  # count date where precipitation  is more than 0.0

    for m in number_of_days:
        with rasterio.open(f'{SATCKED_FILES_CURRENT_DIR}{m}.tif', 'w', **ras_meta) as dst:
            dst.write(np.rint(pd.DataFrame(number_of_days[m].tolist()).astype(
                'float32').to_numpy()/len(years)), 1)


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
        print('1/6- Collecting .tif images links from https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p25/')
        # # launch concurent dowload of all the .tif files selected
        print('2/6- Dowloading the .tif files')
        concurrent_file_downloader(files)
        # dowload aoi file
        print('3/6- Loading aoi shape file')
        aoishapes = aoi_shapefile_reader(aoifilepath)
        # clipping or maksing.  The files are stored in MASKED_FILES_DIR
        print('4/6- Masking the .tif files with the aoi polygon')
        #concurrent_masking(aoishapes, years)

        # Calculate number of raining days
        print('5/6- Calculating the rainy days monthly averages')

        # Generate the stacked files in SATCKED_FILES_DIR

        print(
            f'6/6- Generating the stacked files in {SATCKED_FILES_CURRENT_DIR}')
        stack_rasters(years)
        print(
            f'Your stacked files are available here {SATCKED_FILES_CURRENT_DIR} ')
        print('Monthly rainy days average:')

        # delete_all_downloaded_files(DOWNLOADS_DIR_TIF)
    else:
        print('The year(s) you have chosen are not part of the available data')
    print('... deleting the downloaded .tif files')
