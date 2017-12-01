# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:56:05 2017

@author: Kyle
"""

#_, newFolderName, gpuI, thisTask, dataType, modelName, dataset, learningRate 

import os
import shutil
from sys import argv
_, parentFolderName, gpuIndex = argv

thisTask = 'scene'
dataType = 'waveform'
modelName = 'simpleNetCNN'
dataset = 'DCASE4000'
visualSign = '1'
saveSign = 'True'
iterNum = '150'
#gpuIndex = '0'
#parentFolderName = 'ex5_simpleCNN'

while os.path.isdir( parentFolderName ):
    parentFolderName = parentFolderName + '_1'
    print( 'exist' )

os.mkdir( parentFolderName )
shutil.copy( '../testCode.py', parentFolderName ) # copy this file to the new folder
shutil.copy( '../../model/soundNet.py', parentFolderName )
shutil.copy( '../../model/waveCNN.py', parentFolderName )
shutil.copy( '../../model/specCNN.py', parentFolderName )
shutil.copy( '../../model/testNetSimple.py', parentFolderName )
shutil.copy( '../../model/testNetSimpleRNN.py', parentFolderName )
shutil.copy( '../expUtil.py', parentFolderName )
shutil.copy( 'testScript.py', parentFolderName )

for learningRate in [ 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4 ]: 
    folderName = parentFolderName + '/' + str( learningRate )
    command = 'python ' + '../testCode.py ' + folderName + ' ' + gpuIndex + ' ' \
        + thisTask + ' ' + dataType + ' ' + modelName + ' ' + dataset + ' ' \
        + str( learningRate ) + ' ' + iterNum + ' ' + visualSign + ' ' + saveSign \
        + ' ' + parentFolderName
    os.system( command )