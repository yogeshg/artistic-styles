import os
import shutil
import pytz
import datetime as dt

import logging
logger = logging.getLogger(__file__)

def ensureDir(p):
    if(not os.path.isdir(p)):
        logger.info('creating directory: '+p)
        os.makedirs(p)
    return

def getTs():
    return dt.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')

def cleanDir(p):
    logger.info('cleaning directory: '+p)
    shutil.rmtree(p)
    ensureDir(p)
    return

def archiveDir(p):
    ensureDir(ARCHIVE)
    archivePath = os.path.join(ARCHIVE,getTs())
    wd, zd = os.path.split(p)
    logger.info('archiving directory: '+str((wd,zd)) )
    st = shutil.make_archive( archivePath, 'tar', wd, zd)
    logger.info('archived directory: '+str(st) )
    return

WORKINGDIR = os.getcwd()
DATADIR = os.path.join(WORKINGDIR, 'data')
CURRDIR = os.path.join(DATADIR, 'current')
ARCHIVE = os.path.join(DATADIR, 'archive')

def getFilePath(p):
    return os.path.join(CURRDIR, p)

