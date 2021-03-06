function [ ] = batchCheckResult(  )
clc;clear;

task = 'dcase';
basePath = '../experiment';
baseFileList = dir( basePath );

performanceRecord = [ ];

for experimentTypeIndex = 3 :size( baseFileList, 1 )
    experimentTypeName = baseFileList( experimentTypeIndex ).name;
    if strcmp( experimentTypeName( 1 :5 ), task ) == 1
        typeDirectory = [ basePath, '/', experimentTypeName ];
        tempPerformanceRecord = recordPerformance( typeDirectory );
        performanceRecord = [ performanceRecord, tempPerformanceRecord ];
    end
end
   [ rankedPerformanceRecord, rankedNameRecord ] = rankResult( performanceRecord );
end

function [ performanceRecord ] = recordPerformance( directory )
    
    index = 1;
    experimentList = dir( directory );
    for experimentIndex = 3: size( experimentList, 1 )
        subExperimentName = experimentList( experimentIndex ).name;
        subExperimentDirectory = [ directory, '/', subExperimentName ];
        subExperimentList = dir( subExperimentDirectory );
        for subExperimentIndex = 3 :size( subExperimentList, 1 )
            if subExperimentList( subExperimentIndex ).isdir == 1
                settingName = subExperimentList( subExperimentIndex ).name;
                settingDirectory = [ subExperimentDirectory, '/', settingName ];
                if exist( [ settingDirectory, '/', 'folder0/accuracy.csv' ], 'file' ) == 2
                    [ performanceRecord( index ).testAcc, performanceRecord(index).testEpoch, performanceRecord(index).trainAcc, performanceRecord(index).trainEpoch ]...
                        = readEval( [ settingDirectory, '/', 'folder0/accuracy.csv' ] );         
                    performanceRecord( index ).description = settingDirectory;
                    index = index + 1;
                end
            end
        end
    end
end

function [ testAcc, testEpoch, trainAcc, trainEpoch ] = readEval( filePath )
    
    accuracyFile = csvread( filePath );
    onTest = accuracyFile( 1, : );
    onTrain = accuracyFile( 2, : );
    [ testAcc, testEpoch ] = max( onTest );
    [ trainAcc, trainEpoch ] = max( onTrain );
    
    if testAcc == 1 
        disp( 'a' )
    end

end

function [ rankedPerformanceRecord, rankedNameRecord ] = rankResult( performanceRecord )
    rankedPerformanceRecord = nestedSortStruct( performanceRecord, 'testAcc');
    rankedNameRecord = nestedSortStruct( performanceRecord, 'description');
end
