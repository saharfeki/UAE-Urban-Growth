# 🇦🇪 UAE Urban Growth Analysis — 2.0

An automated **remote sensing and machine learning workflow** for land-cover classification and urban growth analysis across the UAE for **2018, 2021, and 2024**.

## 🔄 Workflow

```text
Sentinel-2 Composites
        ↓
Data Preparation
        ↓
OSM + Manual Polygons
        ↓
Label Extraction
        ↓
Spectral Bands + Indices
        ↓
Feature Extraction
        ↓
Random Forest Training
        ↓
Tile-wise Classification
        ↓
Mosaic & Post-Processing
        ↓
Urban Growth Analysis
```

## 🛰️ Features

| Feature | Description |
|---|---|
| B2 | Blue — atmospheric correction & water detection |
| B3 | Green — vegetation assessment |
| B4 | Red — vegetation stress |
| B8 | NIR — vegetation vigor & biomass |
| B11 | SWIR — soil moisture & built-up detection |
| NDVI | Vegetation density |
| NDBI | Built-up area detection |
| MNDWI | Water enhancement |
| BSI | Bare soil & desert detection |
| GHSL | Settlement presence proxy |

## 🌍 Land-Cover Classes

- 🏙️ Urban
- 🌿 Vegetation
- 🏜️ Bare Soil / Desert
- 💧 Water

## 📊 Outputs

The workflow produces:

- Land-cover area comparisons
- Class distribution statistics
- Land-cover change detection
- Urban growth rates

### Results

![Land Cover Area Comparison](outputs/Land%20Cover%20Area%20Comparison%202018%20%202021%20and%202024.png)

![Land Cover Change Detection](outputs/Land%20Cover%20Change%20Detection%202018%20to%202024%20(Full%20Period).png)

![Urban Growth Rates](outputs/Urban%20Growth%20Rates%20by%20Period.png)

## 🛠️ Technologies

**Sentinel-2 · Google Earth Engine · Python · Random Forest · OpenStreetMap · GHSL · Jupyter Notebook**

## 📁 Structure

```text
UAE-Urban-Growth-2018-2024 _2.0/
├── notebooks/
├── outputs/
└── README.md
```


**Sahar Feki · Yessmine Chaabouni**

Geomatics Engineering Students
