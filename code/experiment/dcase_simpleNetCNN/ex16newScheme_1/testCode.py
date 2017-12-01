# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:03:04 2017

Conduct erxperiment on IEMOCAP, three labels: 
    
    96001: emotion(0-4, 5 = other emotions)
    96002: speaker(0-9)
    96003: gender(male=0, female=1)
    

@author: Kyle
"""

import os
from sys import argv
_, newFolderName, gpuI, thisTask, dataType, modelName, dataset, learningRate, inputIterNum, visualSign, inputSaveSign, parentFoldName = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)

import sys
sys.path.append( parentFoldName ) # import from its own directory
import soundNet
import waveCNN
import testNetSimple
import testNetSimpleRNN
import specCNN
import expUtil
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import shutil

#%% creat folder to save model, the code, and model configuration 
while os.path.isdir( newFolderName ):
    newFolderName = newFolderName + '_1'
    print( 'exist' )

os.mkdir( newFolderName )

#shutil.copy( '../testCode.py', newFolderName ) # copy this file to the new folder
#shutil.copy( '../../model/soundNet.py', newFolderName )
#shutil.copy( '../../model/waveCNN.py', newFolderName )
#shutil.copy( '../../model/specCNN.py', newFolderName )
#shutil.copy( '../../model/testNetSimple.py', newFolderName )
#shutil.copy( '../../model/testNetSimpleRNN.py', newFolderName )
#shutil.copy( '../expUtil.py', newFolderName )

# put all configuratation here
#thisTask = 'emotion'
#dataType = 'toyWaveform'
#modelName = 'simpleNetCNN'
#dataset = 'IEMOCAP'

if modelName == 'simpleNetCNN':
    model = testNetSimple.testNet
elif modelName == 'simpleNetRNN':
    model = testNetSimpleRNN.testNetRNN
elif modelName == 'specNet':
    model = specCNN.specCNN
elif modelName == 'soundNet':
    model = soundNet.soundNet
elif modelName == 'waveCNN':
    model = waveCNN.waveCNNBN
else: 
    print( 'model not exist' )
    
#%% adjust the input parameters
iter_num = int( inputIterNum )
learningRate = float( learningRate )
visualSign = int( visualSign )
if inputSaveSign == 'True':
    saveSign = True
elif inputSaveSign == 'False':
    saveSign = False

#%% store the log file
log = thisTask + '\n' + dataType + '\n' + modelName + '\n' + dataset + '\n' + str( learningRate ) + '\n' + str( iter_num )
with open( newFolderName + '/log.txt' , "w") as text_file:
    text_file.write( log )

    
#%% load data
if ( dataset == 'IEMOCAP' ) or ( dataset == 'ESC' ): 
    for testFolder in [ 0, 1, 2, 3, 4 ]:
        if dataset == 'IEMOCAP':
            trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataIEMOCAP( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        if dataset == 'ESC':
            trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataESC( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        
        newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
        os.mkdir( newFolderNameForThisFolder )
        # train the model
        if dataset == 'IEMOCAP':
            resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                    lr_decay = 0.1, batch_size = 32, learningRate = learningRate, iterationNum = iter_num, \
                                                    modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                    visualSign = visualSign, saveSign = saveSign, dataset = dataset  )
        # add specific task 
        if dataset == 'ESC':
            resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                    lr_decay = 0.1, batch_size = 8, learningRate = learningRate, iterationNum = iter_num, \
                                                    modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                    visualSign = visualSign, saveSign = saveSign, dataset = dataset, task = thisTask )
        
#%% load data of DCASE, only have 4 folds
if dataset == 'DCASE': 
    for testFolder in [ 0 ]:
        trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataDCASE( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        
        newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
        os.mkdir( newFolderNameForThisFolder )
        
        # train the model
        resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                lr_decay = 0.1, batch_size = 8, learningRate = learningRate, iterationNum = iter_num, \
                                                modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                visualSign = visualSign, saveSign = saveSign, dataset = dataset, task = thisTask )
        
#%% load data of DCASE, only have 4 folds
if dataset == 'DCASE16000': 
    for testFolder in [ 0 ]:
        trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataDCASE16000( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        
        newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
        os.mkdir( newFolderNameForThisFolder )
        
        # train the model
        resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                lr_decay = 0.1, batch_size = 24, learningRate = learningRate, iterationNum = iter_num, \
                                                modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                visualSign = visualSign, saveSign = saveSign, dataset = dataset, task = thisTask )

#%% load data of DCASE, only have 4 folds
if dataset == 'DCASE8000': 
    for testFolder in [ 0 ]:
        trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataDCASE8000( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        
        newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
        os.mkdir( newFolderNameForThisFolder )
        
        # train the model
        resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                lr_decay = 0.1, batch_size = 32, learningRate = learningRate, iterationNum = iter_num, \
                                                modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                visualSign = visualSign, saveSign = saveSign, dataset = dataset, task = thisTask )
        
#%% load data of DCASE, only have 4 folds
if dataset == 'DCASE4000': 
    for testFolder in [ 0 ]:
        trainFeature, trainLabel, testFeature, testLabel = expUtil.loadDataDCASE4000( testFolder = testFolder, testTask = thisTask, \
                                                                            precision = 'original', sampleRate = 16000, dataType = dataType )
        
        newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
        os.mkdir( newFolderNameForThisFolder )
        
        # train the model
        resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = iter_num, \
                                                lr_decay = 0.1, batch_size = 32, learningRate = learningRate, iterationNum = iter_num, \
                                                modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, \
                                                visualSign = visualSign, saveSign = saveSign, dataset = dataset, task = thisTask )


