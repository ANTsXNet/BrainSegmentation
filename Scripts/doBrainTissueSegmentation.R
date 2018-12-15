library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 4 )
  {
  helpMessage <- paste0( "Usage:  Rscript doHippoEcSegmentation.R",
    " inputFile inputMaskFile outputFilePrefix reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  inputMaskFileName <- args[2]
  outputFilePrefix <- args [3]
  reorientTemplateFileName <- args[4]
  }

classes <- c( "Background", "Csf", "GrayMatter", "WhiteMatter",
  "DeepGrayMatter", "BrainStem", "Cerebellum"  )
numberOfClassificationLabels <- length( classes )
resampledImageSize <- c( 112, 160, 112 )
segmentationLabels <- seq_len( numberOfClassificationLabels ) - 1
numberOfFiltersAtBaseLayer <- 16

imageMods <- c( "T1" )
channelSize <- length( imageMods )

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, dropoutRate = 0.0,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFiltersAtBaseLayer,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- getPretrainedNetwork( "brainSegmentation" )
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
mask <- antsImageRead( inputMaskFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template and cropping to mask." )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
warpedMask <- applyAntsrTransformToImage( xfrm, mask, reorientTemplate,
  interpolation = "nearestNeighbor" )
warpedMask <- iMath( warpedMask, "MD", 3 )
warpedCroppedImage <- cropImage( warpedImage, warpedMask, 1 )
originalCroppedSize <- dim( warpedCroppedImage )
warpedCroppedImage <- resampleImage( warpedCroppedImage,
  resampledImageSize, useVoxels = TRUE )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

batchX <- array( data = as.array( warpedCroppedImage ),
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, warpedCroppedImage )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Renormalize to native space" )
startTime <- Sys.time()

zeroArray <- array( data = 0, dim = dim( warpedImage ) )
zeroImage <- as.antsImage( zeroArray, reference = warpedImage )

probabilityImages <- list()
for( i in seq_len( numberOfClassificationLabels ) )
  {
  probabilityImageTmp <- resampleImage( probabilityImagesArray[[1]][[i]],
    originalCroppedSize, useVoxels = TRUE )
  probabilityImageTmp <- decropImage( probabilityImageTmp, zeroImage )
  probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImageTmp, image )
  }

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Writing", outputFilePrefix )
startTime <- Sys.time()

probabilityImageFiles <- c()
for( i in seq_len( numberOfClassificationLabels - 1 ) )
  {
  probabilityImageFiles[i] <- paste0( outputFilePrefix, classes[i+1], ".nii.gz" )
  antsImageWrite( probabilityImages[[i+1]], probabilityImageFiles[i] )
  }

probabilityImagesMatrix <- imagesToMatrix( probabilityImageFiles, mask )
segmentationVector <- apply( probabilityImagesMatrix, FUN = which.max, MARGIN = 2 )
segmentationImage <- makeImage( mask, segmentationVector )
antsImageWrite( segmentationImage, paste0( outputFilePrefix, "Segmentation.nii.gz" ) )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
