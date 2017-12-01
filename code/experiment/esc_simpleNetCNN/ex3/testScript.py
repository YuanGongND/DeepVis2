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
dataset = 'ESC'
visualSign = '1'
saveSign = 'True'
iterNum = '100'
#gpuIndex = '0'
#parentFolderName = 'ex5_simpleCNN'

while os.path.isdir( parentFolderName ):
    parentFolderName = parentFolderName + '_1'
    print( 'exist' )

os.mkdir( parentFolderName )
shutil.copy( 'testScript.py', parentFolderName )

for learningRate in [ 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 10e-4 ]: 
    folderName = parentFolderName + '/' + str( learningRate )
    command = 'python ' + '../testCode.py ' + folderName + ' ' + gpuIndex + ' ' \
        + thisTask + ' ' + dataType + ' ' + modelName + ' ' + dataset + ' ' \
        + str( learningRate ) + ' ' + iterNum + ' ' + visualSign + ' ' + saveSign
    os.system( command )