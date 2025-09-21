# lulc-analysis
Description: Examining urbanization effects on landmass using LULC classification, Machine Learning, and Remote Sensing techniques in Dehradun, Uttarakhand
# Urbanization Impact Analysis Using LULC Classification

## 📋 Project Overview
This project examines the effects of urbanization on landmass in Dehradun, Uttarakhand using Land Use Land Cover (LULC) classification, Machine Learning, and Remote Sensing techniques. The study analyzes changes between 2018 and 2024 using Sentinel-2 satellite imagery.

## 🎯 Key Results
- **Overall Accuracy**: 87.60% (2018), 89.66% (2024)
- **Kappa Coefficient**: 0.8412 (2018), 0.8689 (2024)
- **Urban Area Growth**: 657.69 → 809.04 hectares (+23.0%)
- **Water Bodies Increase**: 29.84 → 115.09 hectares (+285.6%)

## 🛠️ Technologies Used
- Google Earth Engine (JavaScript)
- Sentinel-2 Satellite Imagery
- Random Forest Machine Learning
- Remote Sensing & GIS

## 🗺️ Study Area
Dehradun district, Uttarakhand, India covering Lachhiwala range, Ramgarh range, River Song, and Doiwala/Teliwala/Bullawala regions.

## 📊 Land Cover Classes
1. **Water Bodies** (Blue)
2. **Agriculture** (Yellow)  
3. **Forest** (Green)
4. **Barren Land** (Grey)
5. **Urban Area** (Red)

## 🚀 How to Use
1. Open Google Earth Engine Code Editor
2. Import the code from `code/lulc_classification.js`
3. Define your Region of Interest (ROI)
4. Run the classification analysis

## 📁 Project Structure
- `code/`: Google Earth Engine JavaScript files
- `data/`: Training points and datasets
- `images/`: Maps and visualizations
- `results/`: Accuracy metrics and statistics
- `docs/`: Project documentation

