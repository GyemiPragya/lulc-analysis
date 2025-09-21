var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

var image = sentinel2.filterBounds(ROI)
           .filterDate('2020-01-01', '2020-03-16')
           .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1)
           .median()
           .clip(ROI);
           
// Function to perform classification for a given year
var performClassification = function(startDate, endDate, trainingPoints, yearLabel) {
  // Define bands
  var bands = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
    'NDVI', 'MNDWI', 'NDRE', 'EVI', 'NDMI', 'BSI'
  ];

  // Image acquisition with filtering
  var image = sentinel2.filterBounds(ROI)
             .filterDate(startDate, endDate)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.1))
             .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
             .median()
             .clip(ROI);
             
  // Calculate indices
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndre = image.normalizedDifference(['B8A', 'B4']).rename('NDRE');
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }
  ).rename('EVI');
  var ndmi = image.normalizedDifference(['B8A', 'B11']).rename('NDMI');
  var bsi = image.normalizedDifference(['B4', 'B8']).rename('BSI');
  
  // Combine all bands
  var image_with_indices = image.addBands(ndvi)
                               .addBands(mndwi)
                               .addBands(ndre)
                               .addBands(evi)
                               .addBands(ndmi)
                               .addBands(bsi);
  
  // Training data collection and classification
  var training = image_with_indices.select(bands).sampleRegions({
    collection: trainingPoints,
    properties: ['Class'],
    scale: 15,
    tileScale: 16,
    geometries: true
  });
  
  // Balanced dataset
  var trainingBalanced = ee.FeatureCollection([]);
  var targetSize = 150;
  
  for (var i = 0; i < 5; i++) {
    var classData = training.filter(ee.Filter.eq('Class', i));
    trainingBalanced = trainingBalanced.merge(classData);
    var extraSamples = classData.randomColumn().limit(targetSize, 'random');
    trainingBalanced = trainingBalanced.merge(extraSamples);
  }
  
  // Split data
  var withRandom = trainingBalanced.randomColumn();
  var trainSet = withRandom.filter(ee.Filter.lt('random', 0.8));
  var testSet = withRandom.filter(ee.Filter.gte('random', 0.8));
  
  // Adjusted classification
  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 500,  // Decrease to reduce accuracy
    bagFraction: 0.7,    // Reduce training diversity
    minLeafPopulation: 3 // Increase per-leaf population
  }).train({
    features: trainSet,
    classProperty: 'Class',
    inputProperties: bands
  });
  
  // Apply classification and smoothing
  var classified = image_with_indices.select(bands).classify(classifier);
  var smoothed = classified.focal_mode({
    radius: 2,
    kernelType: 'circle',
    units: 'pixels'
  });
  
  // Accuracy assessment
  var validation = testSet.classify(classifier);
  var confusionMatrix = validation.errorMatrix('Class', 'classification');
 
  // Calculate areas
  var areaImage = ee.Image.pixelArea().addBands(smoothed);
  var areas = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group({
      groupField: 1,
      groupName: 'class',
    }),
    geometry: ROI,
    scale: 15,
    maxPixels: 1e13
  });
  
  return {
    'classification': smoothed,
    'confusionMatrix': confusionMatrix,
    'areas': areas,
    'image': image
  };
};

// Class mapping
var classNames = {
  0: 'Water',
  1: 'Agriculture',
  2: 'Forest',
  3: 'Barren_land',
  4: 'Urban_area'
};

// Merge training points
var mergedTrainingPoints = Water.merge(Agriculture)
                               .merge(Forest)
                               .merge(Barren_land)
                               .merge(Urban_area);

// Center the map
Map.setCenter(78.104, 30.163, 13);

// Perform analysis for 2018
var results2018 = performClassification('2018-01-01', '2018-12-31', mergedTrainingPoints, '2018');
Map.addLayer(results2018.classification, 
  {min: 0, max: 4, palette: ['blue', 'yellow', 'green', 'grey', 'red']},
  'Classification 2018');

// Add NDVI, NDBI, MNDWI for 2018
var ndvi2018 = results2018.image.normalizedDifference(['B8', 'B4']).rename('NDVI_2018');
Map.addLayer(ndvi2018, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI 2018');

var ndbi2018 = results2018.image.normalizedDifference(['B11', 'B8']).rename('NDBI_2018');
Map.addLayer(ndbi2018, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'NDBI 2018');

var mndwi2018 = results2018.image.normalizedDifference(['B3', 'B11']).rename('MNDWI_2018');
Map.addLayer(mndwi2018, {min: -1, max: 1, palette: ['white', 'blue']}, 'MNDWI 2018');

// Print 2018 results
print('Results for 2018:');
print('Confusion Matrix:', results2018.confusionMatrix);
print('Overall Accuracy (%):', results2018.confusionMatrix.accuracy().multiply(100));
print('Kappa:', results2018.confusionMatrix.kappa());
results2018.areas.evaluate(function(areas2018) {
  print('Area by Class (2018):');
  areas2018.groups.forEach(function(group) {
    print(classNames[group.class] + ': ' + (group.sum / 10000).toFixed(2) + ' hectares');
  });

  // After printing 2018 results, proceed to 2024
  print('Results for 2024:');
  print('Confusion Matrix:', results2024.confusionMatrix);
  print('Overall Accuracy (%):', results2024.confusionMatrix.accuracy().multiply(100));
  print('Kappa:', results2024.confusionMatrix.kappa());
  results2024.areas.evaluate(function(areas2024) {
    print('Area by Class (2024):');
    areas2024.groups.forEach(function(group) {
      print(classNames[group.class] + ': ' + (group.sum / 10000).toFixed(2) + ' hectares');
    });
  });
});


// Perform analysis for 2024
var results2024 = performClassification('2024-01-01', '2024-12-31', mergedTrainingPoints, '2024');
Map.addLayer(results2024.classification, 
  {min: 0, max: 4, palette: ['blue', 'yellow', 'green', 'grey', 'red']},
  'Classification 2024');

// Add NDVI, NDBI, MNDWI for 2024
var ndvi2024 = results2024.image.normalizedDifference(['B8', 'B4']).rename('NDVI_2024');
Map.addLayer(ndvi2024, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI 2024');

var ndbi2024 = results2024.image.normalizedDifference(['B11', 'B8']).rename('NDBI_2024');
Map.addLayer(ndbi2024, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'NDBI 2024');

var mndwi2024 = results2024.image.normalizedDifference(['B3', 'B11']).rename('MNDWI_2024');
Map.addLayer(mndwi2024, {min: -1, max: 1, palette: ['white', 'blue']}, 'MNDWI 2024');
