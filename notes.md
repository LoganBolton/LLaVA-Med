## Data augmentation

Want to generate a new version of `data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json` with augmented, cropped data. 

Update `data/augment/adjust_crop.py` to be able to do everything. 

**goals**
- The same images as the original chest CT scans, but with 5 zoomed in images per original base image
- A new metadata file that is cloned from the original CT Scan json. 
    - Should have a different metadata attribute called `zoom`
