# Import Packages and setup
from __future__ import print_function, division
import os
from collections import OrderedDict
import logging

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# Make the plots look pretty
matplotlib.rc('font',size=13)
matplotlib.rc('legend', numpoints=1)
matplotlib.rc('image', cmap='inferno')
matplotlib.rc('image', interpolation='none')
# Use the same color cylcer as Seaborn
from cycler import cycler
color_cycle = [u'#4c72b0', u'#55a868', u'#c44e52', u'#8172b2', u'#ccb974', u'#64b5cd']
matplotlib.rcParams['axes.prop_cycle'] = cycler("color", color_cycle)
from astropy.table import Table as ApTable

import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.afw.image import PARENT
from lsst.utils import getPackageDir
import lsst.log as log
import lsst.meas.deblender
from lsst.meas.deblender import proximal, display, sim, baseline
import lsst.meas.deblender.utils as debUtils
from lsst.daf.persistence import Butler


logger = logging.getLogger("lsst.meas.deblender")
logger.setLevel(logging.INFO)
log.setLevel("", log.INFO)
plogger = logging.getLogger("proxmin")
plogger.setLevel(logging.INFO)
dlogger = logging.getLogger("deblender")
dlogger.setLevel(logging.INFO)

def loadCatalogs(filters, path, filename="catalog"):
    catalogs = OrderedDict()
    for f in filters:
        catalogs[f] = afwTable.SourceCatalog.readFits(os.path.join(path, "{0}_{1}.fits".format(filename,f)))
    return catalogs

def getAllFlux(catalogs, filters):
    flux = OrderedDict()
    f = filters[0]
    N = len(catalogs[f])
    sed = np.zeros((N,len(filters)))
    for f in filters:
        flux[f] = np.zeros((N,))
    for idx in range(N):
        if catalogs[f][idx]["parent"]==0:
            continue
        for fidx, f in enumerate(filters):
            flux[f][idx] = np.sum(catalogs[f][idx].getFootprint().getImageArray())
            sed[idx,fidx] = flux[f][idx]
        minFlux = np.min(sed[idx,:])
        boloFlux = np.sum(np.abs(sed[idx,:]))
        if boloFlux>0:
            sed[idx,:] /= boloFlux
    return flux, sed

def buildAllTables(catalogs, flux, sed, filters):
    from astropy.table import Table as ApTable
    f = filters[0]
    ids = catalogs[f]["id"]
    parents = catalogs[f]["parent"]
    allFlux = [flux[f] for f in filters]
    footprints = [[src.getFootprint() for src in catalogs[f]] for f in filters]
    x = np.zeros((len(ids),))
    y = np.zeros((len(ids),))
    for idx, src in enumerate(catalogs[f]):
        peaks = src.getFootprint().getPeaks()
        if len(peaks)>0:
            x[idx] = peaks[0].getFx()
            y[idx] = peaks[0].getFy()
        else:
            x[idx] = np.nan
            y[idx] = np.nan
    names = (["id", "x", "y", "parent", "sed"] +
             ["{0}_footprint".format(f) for f in filters] +
             ["{0}_flux".format(f) for f in filters])
    return ApTable([ids, x, y, parents, sed]+footprints+allFlux, names=tuple(names))

def getCoords(catalog):
    x = np.zeros((len(catalog),))
    y = np.zeros((len(catalog),))
    for idx, src in enumerate(catalog):
        peaks = src.getFootprint().getPeaks()
        if len(peaks)>0:
            x[idx] = peaks[0].getFx()
            y[idx] = peaks[0].getFy()
        else:
            x[idx] = np.nan
            y[idx] = np.nan
    return x,y

def getFlux(catalog):
    flux = np.zeros((len(catalog,)))
    for idx in range(len(catalog)):
        if catalog[idx]["parent"]==0:
            continue
        flux[idx] = np.sum(catalog[idx].getFootprint().getImageArray())
    return flux

def buildTable(catalog):
    ids = catalog["id"]
    parents = catalog["parent"]
    x,y = getCoords(catalog)
    flux = getFlux(catalog)
    footprints = [src.getFootprint() for src in catalog]
    return ApTable([ids, x, y, parents, flux, footprints], names=('id', 'x', 'y', 'parent', 'flux', 'footprint'))

from scipy.spatial import cKDTree as KDTree
def matchCatalog(refTable, tables):
    refCoords = np.array(list(zip(refTable["x"], refTable["y"])))
    matches = np.ones((len(refTable,)), dtype=bool)
    for f, tbl in tables.items():
        tbl = tbl[tbl["parent"]!=0]
        coords = np.array(list(zip(tbl["x"], tbl["y"])))
        kdtree = KDTree(coords)
        d2, idx = kdtree.query(refCoords)
        tables[f] = tbl[idx]
        matches = matches & (d2<=2.00001)
    return matches

def getSed(tables, filters):
    f = filters[0]
    N = len(tables[f])
    sed = np.zeros((N,len(filters)))
    for idx in range(N):
        for fidx, f in enumerate(filters):
            sed[idx,fidx] = np.sum(tables[f]["flux"])
        minFlux = np.min(sed[idx,:])
        boloFlux = np.sum(np.abs(sed[idx,:]))
        if boloFlux>0:
            sed[idx,:] /= boloFlux
    return sed

def matchAllCatalogs(refTable, tables, filters):
    matches = matchCatalog(refTable, tables)
    for f in tables:
        tables[f] = tables[f][matches]
    newResult = refTable[matches]
    x = np.mean([tbl["x"] for f, tbl in tables.items()], axis=0)
    y = np.mean([tbl["y"] for f, tbl in tables.items()], axis=0)
    sed = getSed(tables, filters)
    allFlux = [np.array(tbl["flux"]) for f,tbl in tables.items()]
    names = ["x", "y", "sed"] + ["{0}_flux".format(f) for f in tables]
    return ApTable([x,y, sed]+allFlux, names=tuple(names)), matches

def loadAllTables(filters, newpath, oldpath, dataPath, patch, tract):
    logger.info("Loading new catalog")
    newCats = loadCatalogs(filters, newpath, "template")
    logger.info("Loading new flux conserved catalog")
    newCats2 = loadCatalogs(filters, newpath)
    logger.info("Loading old catalog")
    oldCats = loadCatalogs(filters, oldpath)
    logger.info("Building astropy tables")
    newFlux, newSed = getAllFlux(newCats, filters)
    newFlux2, newSed2 = getAllFlux(newCats2, filters)
    newTable = buildAllTables(newCats, newFlux, newSed, filters)
    newTable2 = buildAllTables(newCats2, newFlux2, newSed2, filters)
    newTable = newTable[(newTable["parent"]!=0) & ~np.isnan(newTable["x"])]
    newTable2 = newTable2[(newTable2["parent"]!=0) & ~np.isnan(newTable2["x"])]
    oldTables = OrderedDict([
        (f, buildTable(oldCats[f])) for f in filters
    ])
    logger.info("matching results")
    oldTable, matches = matchAllCatalogs(newTable, oldTables, filters)
    matchedNew = newTable[matches]
    matchedNew2 = newTable2[matches]
    
    logger.info("loading calexps")
    butler = Butler(inputs=dataPath)
    calexp = OrderedDict()
    for f in filters:
        calexp[f] = butler.get('deepCoadd_calexp', patch=patch, filter="HSC-"+f, tract=tract)
    return oldTable, matchedNew, matchedNew2, calexp, newCats

def showComparison(calexp, catalog, table, parent, filters, outliers=None, fidxs=None):
    from lsst.afw.display.displayLib import getZScale
    if fidxs is None:
        fidxs = [3,2,1]
    src = catalog["R"][catalog["G"]["id"]==parent][0]
    img = calexp["R"][src.getFootprint().getBBox(), PARENT].image.array
    vmin, vmax = getZScale(calexp["R"][src.getFootprint().getBBox(), PARENT].image)
    print("vmin: {0}, vmax: {1}".format(vmin, vmax))
    xmin = src.getFootprint().getBBox().getMinX()
    ymin = src.getFootprint().getBBox().getMinY()
    fig = plt.figure(figsize=(16,12))
    ax2 = fig.add_subplot(1,2,1)
    ax3 = fig.add_subplot(1,2,2)
    display.plotColorImage(Q=10, calexps=[calexp[f][src.getFootprint().getBBox(), PARENT] for f in filters],
                           ax=ax2, show=False, filterIndices=fidxs, vmin=vmin, vmax=vmax)

    model = displayModel(table[table["parent"]==parent], catalog["G"], filters)
    display.plotColorImage(model, Q=10, ax=ax3, show=False, filterIndices=fidxs, vmin=vmin, vmax=vmax)
    for n, src in enumerate(table[table["parent"]==parent]):
        ax2.plot(src["x"]-xmin, src["y"]-ymin, 'cx', mew=2)
        ax3.plot(src["x"]-xmin, src["y"]-ymin, 'cx', mew=2)
        if outliers is not None and outliers[table["parent"]==parent][n]==True:
            c = "red"
        else:
            c = "cyan"
        ax2.text(src["x"]-xmin, src["y"]-ymin, str(n), color=c)
        ax3.text(src["x"]-xmin, src["y"]-ymin, str(n), color=c)
    
    plt.tight_layout()
    plt.show()

def getNewColorFootprint(src):
    _img = afwImage.ImageF(src["footprint"].getBBox())
    src["footprint"].insert(_img)
    img = _img.array
    img = img/src["sed"][0]
    colorImg = np.zeros((len(src["sed"]), img.shape[0], img.shape[1]), dtype=img.dtype)
    for s, flux in enumerate(src["sed"]):
        colorImg[s] = flux*img
    return colorImg

def displayModel(children, catalog, filters):
    parent = catalog[catalog["id"]==children[0]["parent"]][0]
    img = afwImage.ImageF(parent.getFootprint().getBBox())
    colorImg = np.zeros((len(children[0]["sed"]), img.array.shape[0], img.array.shape[1]), dtype=img.array.dtype)
    for src in children:
        for fidx, f in enumerate(filters):
            fp = src[f+"_footprint"]
            _img = afwImage.ImageF(img.getBBox())
            fp.insert(_img)
            colorImg[fidx] += _img.array
    return colorImg

def olddisplayModel(children, catalog):
    parent = catalog[catalog["id"]==children[0]["parent"]][0]
    _img = afwImage.ImageF(parent.getFootprint().getBBox())
    img = _img.array
    colorImg = np.zeros((len(children[0]["sed"]), img.shape[0], img.shape[1]), dtype=img.dtype)
    for src in children:
        __img = afwImage.ImageF(_img.getBBox())
        __img.array /= src["sed"][0]
        src["footprint"].insert(__img)
        for s,flux in enumerate(src["sed"]):
            colorImg[s] += flux*__img.array/src["sed"][0]
    print(np.max(colorImg))
    return colorImg

def compareTotalFlux(tbl1, tbl2, tbl1Name, tbl2Name, filters):
    fig = plt.figure(figsize=(16,8))

    for fidx, f in enumerate(filters):
        ax = fig.add_subplot(2,3,fidx+1)
        x = tbl1["{0}_flux".format(f)]
        y = tbl2["{0}_flux".format(f)]
        maxFlux = max(np.max(x), np.max(y))
        maxFlux = min(maxFlux, 10000)
        tenPct = .1*maxFlux
        ax.plot(x, y, '.')
        ax.plot([0,maxFlux], [0,maxFlux])
        ax.plot([0,maxFlux], [tenPct,maxFlux+tenPct], '--', color='#c44e52')
        ax.plot([0,maxFlux], [-tenPct,maxFlux-tenPct], '--', color='#c44e52')
        ax.set_title("{0}-Band Total Flux".format(f))
        ax.set_xlabel(tbl1Name)
        ax.set_ylabel(tbl2Name)
        ax.set_xlim([0,maxFlux])
        ax.set_ylim([0,maxFlux])
    plt.tight_layout()
    plt.show()

def compareSeds(tbl1, tbl2, f, tbl1Name, tbl2Name, filters):
    diff = np.zeros((len(tbl1),))
    for idx in range(len(tbl1)):
        diff[idx] = np.sum(np.abs(tbl1[idx]["sed"]-tbl2[idx]["sed"]))/len(filters)
    plt.plot(tbl1["{0}_flux".format(f)], diff, '.')
    plt.title("sum(abs({0} - {1}))/# Bands".format(tbl1Name, tbl2Name))
    plt.xlabel("{0} {1}-Band Magnitude".format(tbl1Name, f))
    plt.ylabel("Absolute SED Difference")
    plt.show()

def getOverlap(parents, newCats, matchedNew):
    overlap = np.zeros((len(parents),))
    for p, parent in enumerate(parents):
        fp = newCats["G"][newCats["G"]["id"]==parent][0].getFootprint()
        children = matchedNew[matchedNew["parent"]==parent]
        _img = afwImage.ImageF(fp.getBBox())
        _overlap = np.zeros((_img.array.shape))
        for i in range(len(children)-1):
            img1 = afwImage.ImageF(fp.getBBox())
            img2 = afwImage.ImageF(fp.getBBox())
            src1 = children[i]
            src2 = children[i+1]
            src1["R_footprint"].insert(img1)
            src2["R_footprint"].insert(img2)
            _overlap += img1.array*img2.array
        overlap[p] = np.sum(_overlap)
    return overlap