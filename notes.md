## Data augmentation

Want to generate a new version of `data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json` with augmented, cropped data. 

Update `data/augment/adjust_crop.py` to be able to do everything. 

**goals**
- The same images as the original chest CT scans, but with 5 zoomed in images per original base image
- A new metadata file that is cloned from the original CT Scan json. 
    - Should have a different metadata attribute called `zoom`


# Project goals

## Zoom
Do slight zooming in and see if that influences the image at all

# Results

## Zoom Settings
In CT scan dataset, zoom was 99, 97, 95, 93, 91

## LLava-Med-1.5-7B
|Dataset| Original | Zoomed 
|-------|----------|-------|
| Chest CT Scan| 0.2859 (n=871)| 0.3017 (n=4355)| 

### Chest CT Scan (n=871)
| Zoom Level | Accuracy |
|------------|----------|
| 0.91       | 0.2974   |
| 0.93       | 0.3031   |
| 0.95       | 0.3020   |
| 0.97       | 0.2974   |
| 0.99       | 0.3088   |
| 1.00       | 0.2859   |
