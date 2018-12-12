unetImageBatchGenerator <- function( batchSize = 32L,
                                     resampledImageSize = c( 64, 64, 64 ),
                                     segmentationLabels = NA,
                                     segmentationLabelWeights = NA,
                                     doRandomHistogramMatching = FALSE,
                                     reorientImage = NA,
                                     sourceImageList = NULL,
                                     sourceMaskList = NULL,
                                     segmentationList = NULL,
                                     sourceTransformList = NULL,
                                     referenceImageList = NULL,
                                     referenceTransformList = NULL,
                                     outputFile = NULL )
{

  if( is.null( sourceImageList ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( sourceMaskList ) )
    {
    stop( "Input brain masks must be specified." )
    }
  if( is.null( segmentationList ) )
    {
    stop( "Input segmentation images must be specified." )
    }
  if( is.null( sourceTransformList ) )
    {
    stop( "Input transforms must be specified." )
    }
  if( is.na( reorientImage ) )
    {
    stop( "No reference image specified." )
    }

  if( ! is.null( outputFile ) )
    {
    cat( "CurrentPassCount,BatchCount,Source,Reference\n", file = outputFile )
    }

  if( is.null( referenceImageList ) || is.null( referenceTransformList ) )
    {
    referenceImageList <- sourceImageList
    referenceTransformList <- sourceTransformList
    }

  pairwiseIndices <- expand.grid( source = 1:length( sourceImageList ),
    reference = 1:length( referenceImageList ) )

  # shuffle the pairs
  pairwiseIndices <-
    pairwiseIndices[sample.int( nrow( pairwiseIndices ) ),]

  # shuffle the source data
  sampleIndices <- sample( length( sourceImageList ) )
  sourceImageList <- sourceImageList[sampleIndices]
  sourceMaskList <- sourceMaskList[sampleIndices]
  segmentationList <- segmentationList[sampleIndices]
  sourceTransformList <- sourceTransformList[sampleIndices]

  # shuffle the reference data
  sampleIndices <- sample( length( referenceImageList ) )
  referenceImageList <- referenceImageList[sampleIndices]
  referenceTransformList <- referenceTransformList[sampleIndices]

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= nrow( pairwiseIndices ) )
      {
      # shuffle the source data
      sampleIndices <- sample( length( sourceImageList ) )
      sourceImageList <- sourceImageList[sampleIndices]
      sourceMaskList <- sourceMaskList[sampleIndices]
      segmentationList <- segmentationList[sampleIndices]
      sourceTransformList <- sourceTransformList[sampleIndices]

      # shuffle the reference data
      sampleIndices <- sample( length( referenceImageList ) )
      referenceImageList <- referenceImageList[sampleIndices]
      referenceTransformList <- referenceTransformList[sampleIndices]

      # shuffle the pairs
      pairwiseIndices <-
        pairwiseIndices[sample.int( nrow( pairwiseIndices ) ),]

      currentPassCount <- 0L
      }

    rowIndices <- currentPassCount + 1L:batchSize

    batchIndices <- pairwiseIndices[rowIndices,]

    batchSourceImages <- sourceImageList[batchIndices$source]
    batchMasks <- sourceMaskList[batchIndices$source]
    batchSegmentations <- segmentationList[batchIndices$source]
    batchTransforms <- sourceTransformList[batchIndices$source]

    batchReferenceImages <- referenceImageList[batchIndices$reference]
    batchReferenceTransforms <- referenceTransformList[batchIndices$reference]

    channelSize <- length( batchSourceImages[[1]] )

    batchX <- array( data = 0, dim = c( batchSize, resampledImageSize, channelSize ) )
    batchY <- array( data = 0, dim = c( batchSize, resampledImageSize ) )

    currentPassCount <<- currentPassCount + batchSize

    pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    for( i in seq_len( batchSize ) )
      {
      setTxtProgressBar( pb, i )

      if( !is.null( outputFile ) )
        {
        cat( currentPassCount - batchSize, i, batchSourceImages[[i]][1], batchReferenceImages[[i]][1],
          file = outputFile, sep = ",", append = TRUE )
        cat( "\n", file = outputFile, append = TRUE )
        }

      sourceChannelImages <- batchSourceImages[[i]]
      sourceImageId <- basename( batchSourceImages[[i]][1] )

      referenceImageId <- basename( batchReferenceImages[[i]][1] )
      referenceImage <- antsImageRead( batchReferenceImages[[i]][1] )

      fixedParameters <- getCenterOfMass( reorientImage  )
      translationParameters <- getCenterOfMass( referenceImage ) - fixedParameters
      reorientTransform <- createAntsrTransform( precision = "float",
        type = "AffineTransform", dimension = reorientImage@dimension )
      setAntsrTransformFixedParameters( reorientTransform, fixedParameters )
      reorientTransformParameters <- getAntsrTransformParameters( reorientTransform )
      reorientTransformParameters[10:12] <- reorientTransformParameters[10:12] +
        translationParameters
      setAntsrTransformParameters( reorientTransform, reorientTransformParameters )
      reorientTransformFile <- tempfile( fileext = ".mat" )
      invisible( writeAntsrTransform( reorientTransform, reorientTransformFile ) )

      referenceXfrm <- batchReferenceTransforms[[i]]
      sourceXfrm <- batchTransforms[[i]]

      transforms <- c( reorientTransformFile )
      boolInvert <- c( FALSE )
      if( length( referenceXfrm$invtransforms ) == 4 )
        {
        transforms <- append( transforms,
          c( referenceXfrm$invtransforms[4], referenceXfrm$invtransforms[3],
             referenceXfrm$invtransforms[2], referenceXfrm$invtransforms[1] ) )
        boolInvert <- append( boolInvert, c( TRUE, FALSE, FALSE, FALSE ) )
        } else {
        transforms <- append( transforms,
          c( referenceXfrm$invtransforms[2], referenceXfrm$invtransforms[1] ) )
        boolInvert <- append( boolInvert, c( FALSE, FALSE ) )
        }

      if( length( sourceXfrm$fwdtransforms ) == 4 )
        {
        transforms <- append( transforms,
          c( sourceXfrm$fwdtransforms[4], sourceXfrm$fwdtransforms[3],
             sourceXfrm$fwdtransforms[2], sourceXfrm$fwdtransforms[1] ) )
        boolInvert <- append( boolInvert, c( TRUE, FALSE, FALSE, FALSE ) )
        } else {
        transforms <- append( transforms,
          c( sourceXfrm$fwdtransforms[2], sourceXfrm$fwdtransforms[1] ) )
        boolInvert <- append( boolInvert, c( TRUE, FALSE ) )
        }

      # cat( "Ref: ", batchReferenceImages[[i]][1], "\n" )
      # for( t in transforms )
      #   {
      #   cat( t, "\n" )
      #   }

      sourceMask <- antsImageRead( batchMasks[[i]], dimension = 3 )
      warpedMask <- antsApplyTransforms( reorientTemplate, sourceMask,
        interpolator = "nearestNeighbor", transformlist = transforms,
        whichtoinvert = boolInvert  )
      warpedMask <- iMath( warpedMask, "MD", 3 )

      # cat( "SourceY: ", batchSegmentations[[i]], "\n" )
      sourceY <- antsImageRead( batchSegmentations[[i]], dimension = 3 )
      warpedImageY <- antsApplyTransforms( reorientTemplate, sourceY,
        interpolator = "genericLabel", transformlist = transforms,
        whichtoinvert = boolInvert  )

      warpedImageY <- cropImage( warpedImageY, warpedMask, 1 )
      warpedImageY <- resampleImage( warpedImageY, resampledImageSize,
        useVoxels = TRUE, interpType = 1 )

      warpedArrayY <- as.array( warpedImageY )

      # warpedCroppedMask <- cropImage( warpedMask, warpedMask, 1 )
      # warpedCroppedMask <- resampleImage( warpedCroppedMask, resampledImageSize,
      #   useVoxels = TRUE, interpType = 1 )
      # antsImageWrite( as.antsImage( as.array( warpedCroppedMask ) ), "arrayMask.nii.gz" )
      # antsImageWrite( as.antsImage( warpedArrayY ), "arrayY.nii.gz" )
      batchY[i,,,] <- warpedArrayY

      # Randomly "flip a coin" to see if we perform histogram matching.

      doPerformHistogramMatching <- FALSE
      if( doRandomHistogramMatching == TRUE )
        {
        doPerformHistogramMatching <- sample( c( TRUE, FALSE ), size = 1 )
        }

      # cat( "Hist = ", doPerformHistogramMatching, "\n" );

      for( j in seq_len( channelSize ) )
        {
        # cat( "SourceX: ", sourceChannelImages[j], "\n" )
        sourceX <- antsImageRead( sourceChannelImages[j], dimension = 3 )

        warpedImageX <- antsApplyTransforms( reorientTemplate, sourceX,
          interpolator = "linear", transformlist = transforms,
          whichtoinvert = boolInvert )

        if( doPerformHistogramMatching )
          {
          warpedImageX <- histogramMatchImage( warpedImageX,
            antsImageRead( batchReferenceImages[[i]][j], dimension = 3 ),
            numberOfHistogramBins = 64, numberOfMatchPoints = 16 )
          }

        warpedImageX <- cropImage( warpedImageX, warpedMask, 1 )
        warpedImageX <- resampleImage( warpedImageX, resampledImageSize,
          useVoxels = TRUE, interpType = 0 )

        warpedArrayX <- as.array( warpedImageX )
        warpedArrayX <- ( warpedArrayX - mean( warpedArrayX ) ) /
          sd( warpedArrayX )

        # antsImageWrite( as.antsImage( warpedArrayX ), "arrayX.nii.gz" )
        # readline( prompt = "Press [enter] to continue\n" )
        # stop( "Want to check." )
        batchX[i,,,,j] <- warpedArrayX
        }
      }
    cat( "\n" )

    if( any( is.na( segmentationLabels ) ) )
      {
      segmentationLabels <- sort( unique( as.vector( batchY ) ) )
      }
    encodedBatchY <- encodeUnet( batchY, segmentationLabels )

    if( ! any( is.na( segmentationLabelWeights ) ) )
      {
      batchWeights <- array( data = 0, dim = dim( batchY ) )
      for( l in seq_len( length( segmentationLabels ) ) )
        {
        batchWeights[which( batchY == segmentationLabels[l] )] <-
          segmentationLabelWeights[l]
        }

      for( j in seq_len( length( segmentationLabels ) ) )
        {
        encodedBatchY[,,,,j] <- encodedBatchY[,,,,j] * batchWeights
        }
      }
    return( list( batchX, encodedBatchY ) )
    }
}
