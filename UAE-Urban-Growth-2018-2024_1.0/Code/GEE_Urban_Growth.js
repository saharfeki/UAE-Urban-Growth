 // =======================
// AOI: UAE bounding box
// =======================
var countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");
var uae = countries.filter(ee.Filter.eq('country_na', 'United Arab Emirates'));
var aoi = uae.geometry();
Map.centerObject(aoi, 7);
// =======================
// FUNCTIONS
// =======================
function maskS2SR(img) {
  var scl = img.select('SCL');
  return img.updateMask(
    scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10))
  );
}
function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndbi = img.normalizedDifference(['B11','B8']).rename('NDBI');
  var mndwi = img.normalizedDifference(['B3','B11']).rename('MNDWI');
  var bui = ndbi.subtract(ndvi).rename('BUI');
  // Desert Urban Index for better desert classification
  var dui = ndbi.multiply(1.5).subtract(ndvi).add(mndwi.multiply(-0.5)).rename('DUI');
  return img.addBands([ndvi, ndbi, mndwi, bui, dui]);
}
function annualComposite(year) {
  var start = ee.Date.fromYMD(year, 10, 1);
  var end = ee.Date.fromYMD(year + 1, 4, 30);
  
  // Get the collection
  var collection = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(aoi)
    .filterDate(start, end)
    .map(maskS2SR);
  
  print(year + ' - Total images before filtering:', collection.size());
  
  // Filter to ONLY get the specific bands we need
  var filtered = collection.map(function(img) {
    // Select ONLY the bands we're going to use
    // This ensures all images have the same bands
    return img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'SCL'])
      .copyProperties(img, ['system:time_start']);
  });
  
  print(year + ' - Images after band selection:', filtered.size());
  
  // Take the median
  var img = filtered.median().clip(aoi).toFloat();
  
  return addIndices(img).set('year', year);
}
// =======================
// TRAINING DATA USING 2018 AS REFERENCE
// =======================
print('=== TRAINING DATA PREPARATION (2018 REFERENCE) ===');
// Get 2018 composite for training
var img2018 = annualComposite(2018);
// Enhanced bands with desert-specific index
var trainingBands = ['B2','B3','B4','B8','B11','NDVI','NDBI','MNDWI','BUI','DUI'];
// Function to collect training samples with validation
function sampleTraining(polygon, classId, className, nSamples) {
  return img2018.select(trainingBands)
    .sample({
      region: polygon,
      scale: 30, // 30m resolution for training
      numPixels: nSamples,
      seed: 42,
      tileScale: 4,
      geometries: false
    })
    .map(function(f) {
      return f.set({
        'class': classId,
        'label': className
      });
    });
}
// Collect samples from your improved polygons
var urbanSamples = sampleTraining(Urban, 0, 'Urban', 800);
var vegSamples = sampleTraining(vegetation, 1, 'Vegetation', 600);
var soilSamples = sampleTraining(BareSoil, 2, 'BareSoil', 1000); // More desert samples
var waterSamples = sampleTraining(Water, 3, 'Water', 400);
// Merge all samples
var allSamples = urbanSamples
  .merge(vegSamples)
  .merge(soilSamples)
  .merge(waterSamples);
print('Total training samples:', allSamples.size());
print('Class distribution:', allSamples.aggregate_histogram('class'));
// Split into training and validation
var split = allSamples.randomColumn('random', 42);
var trainingSet = split.filter(ee.Filter.lt('random', 0.7));
var validationSet = split.filter(ee.Filter.gte('random', 0.7));
print('Training set:', trainingSet.size());
print('Validation set:', validationSet.size());
// =======================
// TRAIN CLASSIFIER
// =======================
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  minLeafPopulation: 5,
  bagFraction: 0.7,
  seed: 42
}).train({
  features: trainingSet,
  classProperty: 'class',
  inputProperties: trainingBands
});
// Validate classifier
var validated = validationSet.classify(classifier);
var confusionMatrix = validated.errorMatrix('class', 'classification');
print('\n=== VALIDATION RESULTS ===');
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());
// Get producer and user accuracy safely
var producerAcc = confusionMatrix.producersAccuracy();
var userAcc = confusionMatrix.consumersAccuracy();
// Check if we can safely get urban accuracy (class 0)
producerAcc.evaluate(function(pa) {
  userAcc.evaluate(function(ua) {
    // Producer Accuracy (Urban = class 0)
    if (pa && pa.length > 0 && pa[0].length > 0) {
      var paUrban = pa[0][0]; // extract number
      print('Producer Accuracy (Urban): ' + paUrban.toFixed(3));
    } else {
      print('Producer Accuracy (Urban): N/A');
    }
    // User Accuracy (Urban = class 0)
    if (ua && ua.length > 0 && ua[0].length > 0) {
      var uaUrban = ua[0][0];
      print('User Accuracy (Urban): ' + uaUrban.toFixed(3));
    } else {
      print('User Accuracy (Urban): N/A');
    }
  });
});
// =======================
// DESERT-OPTIMIZED CLASSIFICATION FUNCTION
// =======================
function classifyYearOptimized(year) {
  var img = annualComposite(year);
  
  // Raw classification
  var classified = img.select(trainingBands).classify(classifier);
  // --- Only two SAFE corrections ---
  // 1) Water always water
  var waterMask = img.select('MNDWI').gt(0.1);
  classified = classified.where(waterMask, 3);
  // 2) Vegetation always vegetation
  var vegMask = img.select('NDVI').gt(0.35);
  classified = classified.where(vegMask, 1);
  // --- Very light spatial smoothing ---
  classified = classified.focal_mode({
    kernel: ee.Kernel.square(1),
    iterations: 1
  });
  return classified
    .clip(aoi)
    .rename('classification')
    .set('year', year);
}
// =======================
// PROCESS YEARS 2018-2024
// =======================
var years = [2018, 2019, 2020, 2021, 2022, 2023, 2024];
print('\n=== PROCESSING YEARS 2018-2024 ===');
var results = [];
// =======================
// Function to process years one by one
// =======================
function processYear(index) {
  if (index >= years.length) {
    calculateFinalResults(results);
    return;
  }
  
  var year = years[index];
  print('Processing ' + year + '...');
  
  var classified = classifyYearOptimized(year);
  
  // =======================
  // Calculate urban area
  // =======================
// =======================
// Calculate urban area (ROBUST VERSION)
// =======================
// Create area image with band name preserved
// =======================
// Calculate urban area (FAST & SAFE VERSION)
// =======================
var urbanAreaImg = classified
  .eq(0)
  .selfMask()
  .multiply(ee.Image.pixelArea());

// Reduce at coarser scale to avoid timeout
var urbanArea = urbanAreaImg.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: aoi,
  scale: 100,        // <<< THIS IS THE KEY FIX (100m)
  maxPixels: 1e13,
  tileScale: 16
});

// Debug print
print('Raw area dict ' + year, urbanArea);

// Extract value safely
urbanArea.evaluate(function(stats) {
  if (!stats) {
    print('⚠️ ' + year + ' stats is null — skipping year');
    processYear(index + 1);
    return;
  }

  var key = Object.keys(stats)[0];
  var area_m2 = stats[key];

  if (area_m2 === null) {
    print('⚠️ ' + year + ' area is null — skipping year');
    processYear(index + 1);
    return;
  }

  var areaKm2 = area_m2 / 1e6;

  results.push({
    year: year,
    urbanArea: areaKm2,
    image: classified
  });

  print('✅ ' + year + ' urban area: ' + areaKm2.toFixed(2) + ' km²');

  processYear(index + 1);
});
}
// =======================
// Start processing
// =======================
processYear(0);
// =======================
// FINAL RESULTS CALCULATION
// =======================
function calculateFinalResults(results) {
  // Sort by year
  results.sort(function(a, b) { 
    return a.year - b.year; 
  });
  
  // Calculate UAE total area
  var uaeTotalArea = aoi.area().divide(1e6).getInfo();
  
  print('\n' + createLine(55));
  print('URBAN GROWTH ANALYSIS - UAE (2018-2024)');
  print(createLine(55));
  print('UAE Total Area: ' + uaeTotalArea.toFixed(2) + ' km²\n');
  
  print('Year | Urban Area (km²) | % of UAE | Growth from 2018');
  print(createDashLine(55));
  
  var base2018 = results[0].urbanArea;
  
  for (var j = 0; j < results.length; j++) {
    var result = results[j];
    var percent = (result.urbanArea / uaeTotalArea * 100).toFixed(2);
    var growth = ((result.urbanArea - base2018) / base2018 * 100).toFixed(1);
    
    // Format output
    var yearStr = result.year.toString();
    var areaStr = result.urbanArea.toFixed(2);
    var percentStr = percent + '%';
    var growthStr = growth + '%';
    
    // Pad strings for alignment
    yearStr = padRight(yearStr, 4);
    areaStr = padLeft(areaStr, 12);
    percentStr = padLeft(percentStr, 9);
    growthStr = padLeft(growthStr, 6);
    
    print(yearStr + ' | ' + areaStr + ' | ' + percentStr + ' | ' + growthStr);
  }
  
  // Summary statistics
  var lastYear = results[results.length - 1];
  var totalGrowth = lastYear.urbanArea - base2018;
  var totalGrowthPct = (totalGrowth / base2018 * 100).toFixed(1);
  var annualAvgGrowth = (totalGrowth / 6).toFixed(2); // 6 years from 2018-2024
  
  print('\n' + createLine(35));
  print('SUMMARY (2018-2024)');
  print(createLine(35));
  print('Urban area in 2018: ' + base2018.toFixed(2) + ' km²');
  print('Urban area in 2024: ' + lastYear.urbanArea.toFixed(2) + ' km²');
  print('Total urban growth: ' + totalGrowth.toFixed(2) + ' km²');
  print('Percentage growth: ' + totalGrowthPct + '%');
  print('Average annual growth: ' + annualAvgGrowth + ' km²/year');
  print('Average annual growth rate: ' + (parseFloat(totalGrowthPct) / 6).toFixed(2) + '%/year');
  
  // Export results
  exportResults(results, uaeTotalArea);
  
  // Add visualizations
 // addVisualizations(results);
}
// Helper functions for formatting
function createLine(length) {
  var line = '';
  for (var i = 0; i < length; i++) line += '=';
  return line;
}
function createDashLine(length) {
  var line = '';
  for (var i = 0; i < length; i++) line += '-';
  return line;
}
function padLeft(str, length) {
  str = str.toString();
  while (str.length < length) str = ' ' + str;
  return str;
}
function padRight(str, length) {
  str = str.toString();
  while (str.length < length) str = str + ' ';
  return str;
}
// =======================
// =======================
// EXPORT FUNCTION (FIXED)
// =======================
function exportResults(results, totalArea) {
  var features = [];
  
  for (var i = 0; i < results.length; i++) {
    var result = results[i];
    
    var growthFrom2018 = result.urbanArea - results[0].urbanArea;
    var growthPercent = (growthFrom2018 / results[0].urbanArea) * 100;
    var percentageOfUAE = (result.urbanArea / totalArea) * 100;
    
    features.push(
      ee.Feature(null, {
        'Year': result.year,
        'Urban_Area_km2': result.urbanArea,
        'Percentage_of_UAE': percentageOfUAE,
        'Growth_from_2018_km2': growthFrom2018,
        'Growth_from_2018_pct': growthPercent
      })
    );
  }
  Export.table.toDrive({
    collection: ee.FeatureCollection(features),
    description: 'UAE_Urban_Growth_2018_2024',
    fileFormat: 'CSV'
  });
  print('✓ Export task created: UAE_Urban_Growth_2018_2024');
}
// =======================
// VISUALIZATIONS
// =======================
function addYearLayers(year) {
  print('Adding visualization layers for', year);
  var img = annualComposite(year);
  // ---------- RGB ----------
  Map.addLayer(
    img.select(['B4','B3','B2']),
    {min: 0, max: 3000, gamma: 1.3},
    year + ' - RGB',
    false
  );
  // ---------- NDVI ----------
  Map.addLayer(
    img.select('NDVI'),
    {min: -0.2, max: 0.5, palette: ['brown','yellow','lightgreen','green','darkgreen']},
    year + ' - NDVI',
    false
  );
  // ---------- NDBI ----------
  Map.addLayer(
    img.select('NDBI'),
    {min: -0.3, max: 0.3, palette: ['blue','white','red']},
    year + ' - NDBI',
    false
  );
  // ---------- BUI ----------
  Map.addLayer(
    img.select('BUI'),
    {min: -0.4, max: 0.4, palette: ['green','white','orange','red']},
    year + ' - BUI',
    false
  );
  // ---------- MNDWI ----------
  Map.addLayer(
    img.select('MNDWI'),
    {min: -0.4, max: 0.4, palette: ['brown','white','lightblue','blue']},
    year + ' - MNDWI',
    false
  );
  // ---------- Classification ----------
  var classified = classifyYearOptimized(year);
  Map.addLayer(
    classified,
    {min: 0, max: 3, palette: ['red','green','tan','blue']},
    year + ' - Land Cover',
    false
  );
  // ---------- Urban Mask ----------
  Map.addLayer(
    classified.eq(0).selfMask(),
    {palette: ['red']},
    year + ' - Urban Mask',
    false
  );
}
var yearsVis = [2018,2019,2020,2021,2022,2023,2024];
yearsVis.forEach(function(y){
  addYearLayers(y);
});
// EXPORT FINAL CLASSIFIED MAPS
// =======================
// 2018 classified map
var classified2018 = classifyYearOptimized(2018);
Export.image.toAsset({
  image: classified2018,        // your final cleaned classification
  description: 'LC_2018_Final',
  assetId: 'users/fekisahar0/LC_2018_Final',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
var classified2024 = classifyYearOptimized(2024);
// 2024 classified map
Export.image.toAsset({
  image: classified2024,
  description: 'LC_2024_Final',
  assetId: 'users/fekisahar0/LC_2024_Final',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
// =======================
// URBAN CHANGE MAP
// =======================
// Urban class = 0 (adjust if different)
var urban2018 = classified2018.eq(0);
var urban2024 = classified2024.eq(0);
// New urban growth (2018–2024)
var newUrban = urban2024.and(urban2018.not()).selfMask();
Export.image.toAsset({
  image: newUrban,
  description: 'Urban_Growth_2018_2024',
  assetId: 'users/chaabouniyessmine3/Urban_Growth_2018_2024',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: newUrban,
  description: 'Urban_Growth_2018_2024_Figure',
  folder: 'Capstone_Results',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});





































// ============================================
// ===== UAE URBAN GROWTH MONITORING DASHBOARD =====
// ============================================

// Clear default UI
ui.root.clear();

// Define color scheme for professional look
var colors = {
  primary: '#0b2c4a',
  secondary: '#1a5a96',
  accent: '#e63946',
  background: '#f8f9fa',
  success: '#2a9d8f',
  warning: '#e9c46a',
  info: '#457b9d'
};

// Create control panel
var controlPanel = ui.Panel({
  style: {
    width: '400px',
    padding: '15px',
    backgroundColor: colors.background
  }
});

// Create map panel
var mapPanel = ui.Map();
mapPanel.setOptions('SATELLITE', {maxZoom: 18});
mapPanel.centerObject(aoi, 8);

// Use SplitPanel for professional layout
var splitPanel = ui.SplitPanel({
  firstPanel: controlPanel,
  secondPanel: mapPanel,
  orientation: 'horizontal',
  wipe: true,
  style: {stretch: 'both'}
});

ui.root.add(splitPanel);

// ============================================
// HEADER SECTION
// ============================================
var headerPanel = ui.Panel({
  widgets: [
    ui.Label('UAE URBAN GROWTH MONITORING DASHBOARD', {
      fontSize: '22px',
      fontWeight: 'bold',
      color: colors.primary,
      margin: '0 0 8px 0'
    }),
    ui.Label('Sentinel-2 Based Land Cover Classification (2018–2024)', {
      fontSize: '14px',
      color: colors.secondary,
      margin: '0 0 4px 0'
    }),
    ui.Label('Capstone Project - Urban Expansion Analysis', {
      fontSize: '12px',
      color: '#666',
      fontStyle: 'italic',
      margin: '0 0 10px 0'
    }),
    ui.Label('─────────────────────────────────────', {
      color: colors.secondary,
      margin: '5px 0 15px 0'
    })
  ],
  layout: ui.Panel.Layout.flow('vertical')
});

controlPanel.add(headerPanel);

// ============================================
// YEAR SELECTOR
// ============================================
controlPanel.add(ui.Label('TEMPORAL ANALYSIS', {
  fontSize: '14px',
  fontWeight: 'bold',
  color: 'white',
  margin: '20px 0 8px 0',
  backgroundColor: colors.primary,
  padding: '6px 10px'
}));

var yearSelect = ui.Select({
  items: ['2018','2019','2020','2021','2022','2023','2024'],
  value: '2024',
  style: {width: '100%', height: '36px', fontSize: '14px'}
});

var yearPanel = ui.Panel({
  widgets: [
    ui.Label('Select Year:', {
      fontWeight: 'bold',
      margin: '0 0 5px 0',
      fontSize: '13px'
    }),
    yearSelect
  ],
  layout: ui.Panel.Layout.flow('vertical')
});

controlPanel.add(yearPanel);

// ============================================
// LAYER CONTROLS
// ============================================
controlPanel.add(ui.Label('─────────────────────────────────────', {
  color: colors.secondary,
  margin: '15px 0 5px 0'
}));

controlPanel.add(ui.Label('LAYER VISUALIZATION', {
  fontSize: '14px',
  fontWeight: 'bold',
  color: 'white',
  margin: '10px 0 8px 0',
  backgroundColor: colors.primary,
  padding: '6px 10px'
}));

// Base Layers
controlPanel.add(ui.Label('Base Layers:', {
  fontWeight: 'bold',
  margin: '15px 0 8px 0',
  fontSize: '13px',
  color: colors.secondary
}));

var cbRGB = ui.Checkbox('RGB Composite', true);
var cbLC = ui.Checkbox('Land Cover Classification', true);
var cbUrban = ui.Checkbox('Urban Mask', true);
var cbGrowth = ui.Checkbox('Urban Growth (2018-2024)', false);

controlPanel.add(cbRGB);
controlPanel.add(cbLC);
controlPanel.add(cbUrban);
controlPanel.add(cbGrowth);

// Spectral Indices
controlPanel.add(ui.Label('Spectral Indices:', {
  fontWeight: 'bold',
  margin: '15px 0 8px 0',
  fontSize: '13px',
  color: colors.secondary
}));

var cbNDVI = ui.Checkbox('NDVI - Vegetation Index', false);
var cbNDBI = ui.Checkbox('NDBI - Built-up Index', false);
var cbBUI = ui.Checkbox('BUI - Built-up Index', false);
var cbMNDWI = ui.Checkbox('MNDWI - Water Index', false);

controlPanel.add(cbNDVI);
controlPanel.add(cbNDBI);
controlPanel.add(cbBUI);
controlPanel.add(cbMNDWI);

// ============================================
// LEGEND SYSTEM
// ============================================
controlPanel.add(ui.Label('─────────────────────────────────────', {
  color: colors.secondary,
  margin: '15px 0 5px 0'
}));

controlPanel.add(ui.Label('LEGEND & COLOR SCHEMES', {
  fontSize: '14px',
  fontWeight: 'bold',
  color: 'white',
  margin: '10px 0 8px 0',
  backgroundColor: colors.primary,
  padding: '6px 10px'
}));

function createLegendSection(title) {
  return ui.Label(title, {
    fontWeight: 'bold',
    margin: '15px 0 8px 0',
    fontSize: '13px',
    color: colors.secondary
  });
}

function legendItem(color, label, description) {
  return ui.Panel({
    widgets: [
      ui.Label('', {
        backgroundColor: color,
        padding: '10px',
        margin: '0 10px 0 0',
        width: '20px',
        height: '20px'
      }),
      ui.Panel({
        widgets: [
          ui.Label(label, {fontWeight: 'bold', fontSize: '12px'}),
          ui.Label(description, {fontSize: '11px', color: '#666'})
        ],
        layout: ui.Panel.Layout.flow('vertical')
      })
    ],
    layout: ui.Panel.Layout.flow('horizontal')
  });
}

// Land Cover Legend
controlPanel.add(createLegendSection('Land Cover Classification'));
controlPanel.add(legendItem('red', 'Urban', 'Built-up areas'));
controlPanel.add(legendItem('green', 'Vegetation', 'Forests, crops, parks'));
controlPanel.add(legendItem('#d2b48c', 'Bare Soil', 'Desert, barren land'));
controlPanel.add(legendItem('blue', 'Water', 'Rivers, lakes, sea'));
controlPanel.add(legendItem('#ff0000', 'New Growth', 'Urban expansion 2018-2024'));

// Spectral Indices Legend
controlPanel.add(createLegendSection('Spectral Indices'));

function createIndexLegend(title, colorsArray, minVal, maxVal, description) {
  var swatchPanel = ui.Panel({
    widgets: [],
    layout: ui.Panel.Layout.flow('horizontal')
  });
  
  var numColors = colorsArray.length;
  var swatchWidth = Math.floor(300 / numColors);
  
  for (var i = 0; i < numColors; i++) {
    swatchPanel.add(ui.Label('', {
      backgroundColor: colorsArray[i],
      width: swatchWidth + 'px',
      height: '25px',
      margin: '0'
    }));
  }
  
  return ui.Panel({
    widgets: [
      ui.Label(title, {fontWeight: 'bold', fontSize: '12px', margin: '0 0 5px 0'}),
      swatchPanel,
      ui.Panel({
        widgets: [
          ui.Label(minVal.toString(), {fontSize: '10px'}),
          ui.Label('', {stretch: 'horizontal'}),
          ui.Label(maxVal.toString(), {fontSize: '10px'})
        ],
        layout: ui.Panel.Layout.flow('horizontal')
      }),
      ui.Label(description, {fontSize: '11px', color: '#666', margin: '5px 0 15px 0'})
    ],
    layout: ui.Panel.Layout.flow('vertical')
  });
}

// Add index legends
controlPanel.add(createIndexLegend(
  'NDVI - Vegetation Index',
  ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen'],
  -0.2, 0.5,
  'Negative: Water/Bare Soil | Positive: Vegetation'
));

controlPanel.add(createIndexLegend(
  'NDBI - Built-up Index',
  ['blue', 'white', 'red'],
  -0.3, 0.3,
  'Negative: Non-built | Positive: Built-up areas'
));

controlPanel.add(createIndexLegend(
  'BUI - Built-up Index',
  ['green', 'white', 'orange', 'red'],
  -0.4, 0.4,
  'Negative: Vegetation | Positive: Built-up intensity'
));

controlPanel.add(createIndexLegend(
  'MNDWI - Water Index',
  ['brown', 'white', 'lightblue', 'blue'],
  -0.4, 0.4,
  'Negative: Land | Positive: Water bodies'
));

// ============================================
// STATISTICS PANEL
// ============================================
controlPanel.add(ui.Label('─────────────────────────────────────', {
  color: colors.secondary,
  margin: '15px 0 5px 0'
}));

controlPanel.add(ui.Label('STATISTICS & METRICS', {
  fontSize: '14px',
  fontWeight: 'bold',
  color: 'white',
  margin: '10px 0 8px 0',
  backgroundColor: colors.primary,
  padding: '6px 10px'
}));

// Create stats display
var statsLabel = ui.Label(
  'Year: 2024\n' +
  '────────────────\n' +
  'Urban Area: Loading...\n' +
  '────────────────\n' +
  'Analysis Scale: 100m\n' +
  'Data Source: Sentinel-2',
  {
    fontSize: '12px',
    color: '#333',
    whiteSpace: 'pre',
    padding: '15px',
    backgroundColor: '#ffffff',
    margin: '10px 0'
  }
);

controlPanel.add(statsLabel);

// ============================================
// MAP CONTROLS
// ============================================
controlPanel.add(ui.Label('─────────────────────────────────────', {
  color: colors.secondary,
  margin: '15px 0 5px 0'
}));

controlPanel.add(ui.Label('MAP CONTROLS', {
  fontSize: '14px',
  fontWeight: 'bold',
  color: 'white',
  margin: '10px 0 8px 0',
  backgroundColor: colors.primary,
  padding: '6px 10px'
}));

// Opacity Control
var opacityLabel = ui.Label('Top Layer Opacity: 80%', {fontSize: '12px', margin: '0 0 5px 0'});
var opacitySlider = ui.Slider(0, 100, 10, 80, function(value) {
  opacityLabel.setValue('Top Layer Opacity: ' + value + '%');
  var layers = mapPanel.layers();
  var count = layers.length().getInfo();
  if (count > 0) {
    layers.get(count - 1).setOpacity(value / 100);
  }
});

controlPanel.add(opacityLabel);
controlPanel.add(opacitySlider);

// Reset View Button
var resetButton = ui.Button({
  label: '↻ Reset View to UAE',
  onClick: function() {
    mapPanel.centerObject(aoi, 8);
  }
});

controlPanel.add(resetButton);

// Layer Visibility Toggle Button
var toggleButton = ui.Button({
  label: '👁️ Toggle All Layers',
  onClick: function() {
    var allOn = cbRGB.getValue() && cbLC.getValue() && cbNDVI.getValue();
    var newState = !allOn;
    
    cbRGB.setValue(newState, true);
    cbLC.setValue(newState, true);
    cbUrban.setValue(newState, true);
    cbNDVI.setValue(newState, true);
    cbNDBI.setValue(newState, true);
    cbBUI.setValue(newState, true);
    cbMNDWI.setValue(newState, true);
  }
});

controlPanel.add(toggleButton);

// ============================================
// FOOTER
// ============================================
controlPanel.add(ui.Label('─────────────────────────────────────', {
  color: colors.secondary,
  margin: '15px 0 5px 0'
}));
controlPanel.add(ui.Label('© Capstone Project 2024', {
  fontSize: '11px',
  color: '#999',
  fontStyle: 'italic',
  margin: '5px 0 0 0'
}));

// ============================================
// ENHANCED UPDATE FUNCTION
// ============================================
function updateMap() {
  mapPanel.layers().reset();
  
  var year = parseInt(yearSelect.getValue());
  var img = annualComposite(year);
  var classified = classifyYearOptimized(year);
  
  // Apply opacity from slider
  var opacity = opacitySlider.getValue() / 100;
  
  // RGB Composite
  if (cbRGB.getValue()) {
    var rgbLayer = img.select(['B4','B3','B2']);
    mapPanel.addLayer(rgbLayer,
      {
        min: 0,
        max: 3000,
        gamma: 1.3,
        opacity: opacity
      },
      '📷 ' + year + ' - RGB Composite'
    );
  }
  
  // Land Cover Classification
  if (cbLC.getValue()) {
    var lcLayer = classified;
    mapPanel.addLayer(lcLayer,
      {
        min: 0,
        max: 3,
        palette: ['red', 'green', '#d2b48c', 'blue'],
        opacity: 0.7
      },
      '🗺️ ' + year + ' - Land Cover'
    );
  }
  
  // Urban Mask
  if (cbUrban.getValue()) {
    var urbanLayer = classified.eq(0).selfMask();
    mapPanel.addLayer(urbanLayer,
      {
        palette: ['red'],
        opacity: 0.5
      },
      '🏙️ ' + year + ' - Urban Area'
    );
  }
  
  // Urban Growth Analysis
  if (cbGrowth.getValue()) {
    var u2018 = classifyYearOptimized(2018).eq(0);
    var u2024 = classifyYearOptimized(2024).eq(0);
    var newUrban = u2024.and(u2018.not()).selfMask();
    
    mapPanel.addLayer(newUrban,
      {
        palette: ['#ff0000'],
        opacity: 0.7
      },
      '📈 Urban Growth (2018-2024)'
    );
  }
  
  // Spectral Indices
  if (cbNDVI.getValue()) {
    var ndviLayer = img.select('NDVI');
    mapPanel.addLayer(ndviLayer,
      {
        min: -0.2,
        max: 0.5,
        palette: ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen'],
        opacity: 0.8
      },
      '🌿 NDVI - Vegetation Index'
    );
  }
  
  if (cbNDBI.getValue()) {
    var ndbiLayer = img.select('NDBI');
    mapPanel.addLayer(ndbiLayer,
      {
        min: -0.3,
        max: 0.3,
        palette: ['blue', 'white', 'red'],
        opacity: 0.8
      },
      '🏢 NDBI - Built-up Index'
    );
  }
  
  if (cbBUI.getValue()) {
    var buiLayer = img.select('BUI');
    mapPanel.addLayer(buiLayer,
      {
        min: -0.4,
        max: 0.4,
        palette: ['green', 'white', 'orange', 'red'],
        opacity: 0.8
      },
      '📊 BUI - Built-up Index'
    );
  }
  
  if (cbMNDWI.getValue()) {
    var mndwiLayer = img.select('MNDWI');
    mapPanel.addLayer(mndwiLayer,
      {
        min: -0.4,
        max: 0.4,
        palette: ['brown', 'white', 'lightblue', 'blue'],
        opacity: 0.8
      },
      '💧 MNDWI - Water Index'
    );
  }
  
  // ============================================
  // UPDATE STATISTICS
  // ============================================
  var urbanAreaImg = classified.eq(0).selfMask().multiply(ee.Image.pixelArea());
  
  var area = urbanAreaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 100,
    maxPixels: 1e13
  });
  
  area.evaluate(function(stats) {
    var key = Object.keys(stats)[0];
    var areaKm2 = stats[key] / 1e6;
    
    // Get UAE total area for percentage calculation
    var uaeAreaKm2 = aoi.area().divide(1e6);
    uaeAreaKm2.evaluate(function(totalArea) {
      var areaPercent = (areaKm2 / totalArea) * 100;
      
      // Update stats display
      statsLabel.setValue(
        'Year: ' + year + '\n' +
        '────────────────\n' +
        'Urban Area: ' + areaKm2.toFixed(2) + ' km²\n' +
        'Percentage: ' + areaPercent.toFixed(2) + '%\n' +
        '────────────────\n' +
        'Analysis Scale: 100m\n' +
        'Data Source: Sentinel-2'
      );
    });
  });
}

// ============================================
// EVENT HANDLERS
// ============================================
// Connect all controls to update function
var controls = [yearSelect, cbRGB, cbLC, cbUrban, cbGrowth, cbNDVI, cbNDBI, cbBUI, cbMNDWI];
controls.forEach(function(control) {
  control.onChange(updateMap);
});

// Add keyboard shortcut for reset
ui.root.onKeyDown(function(event) {
  var key = event.getKey();
  if (key === 'r' || key === 'R') {
    mapPanel.centerObject(aoi, 8);
  }
});

// Initial load
updateMap();