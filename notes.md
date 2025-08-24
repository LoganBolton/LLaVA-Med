## Data augmentation

Want to generate a new version of `data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json` with augmented, cropped data. 

Update `data/augment/adjust_crop.py` to be able to do everything. 

**goals**
- The same images as the original chest CT scans, but with 5 zoomed in images per original base image
- A new metadata file that is cloned from the original CT Scan json. 
    - Should have a different metadata attribute called `zoom`

# Random

- Llava-Med surprisingly consistent
    - Minor discrepancies with answers
    - Does it use some sort of cropping strat? need to reread paper
    - Should analyze fail cases when answers vary across zoom levels


# TODOs

- Need to run with med gemma
- Prolly need to rerun chest ct scan with better eval script
- Run with different contrast levels


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

### Covid-19 tianchi (n=96)
| Zoom Level | Accuracy |
|------------|----------|
| 0.91       | 0.4167   |
| 0.93       | 0.4271   |
| 0.95       | 0.4271   |
| 0.97       | 0.4062   |
| 0.99       | 0.4167   |
| 1.0        | 0.4167   |



/home/log/Github/LLaVA-Med/data/OmniMedVQA/augmented/Images/Chest CT Scan/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000158_zoom_0.99.png
/home/log/Github/LLaVA-Med/data/OmniMedVQA/Images/Chest CT Scan/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000158.png

## Crop
### ct scan
| crop Level | Accuracy | Correct | Total |
|------------|----------|---------|-------|
| 0.91       | 0.2939   | 256     | 871   |
| 0.93       | 0.2985   | 260     | 871   |
| 0.95       | 0.2997   | 261     | 871   |
| 0.97       | 0.2997   | 261     | 871   |
| 0.99       | 0.2928   | 255     | 871   |
| 1.0        | 0.2939   | 256     | 871   |

Agreement Analysis:
Overall Agreement Rate: 0.9277 (92.77%)
Questions with Full Agreement: 808

### Covid-19 Tianchi
| crop Level | Accuracy | Correct | Total |
|------------|----------|---------|-------|
| 0.91       | 0.4062   | 39      | 96    |
| 0.93       | 0.4167   | 40      | 96    |
| 0.95       | 0.4062   | 39      | 96    |
| 0.97       | 0.4062   | 39      | 96    |
| 0.99       | 0.4271   | 41      | 96    |
| 1.0        | 0.4167   | 40      | 96    |

Agreement Analysis:
  Overall Agreement Rate: 0.9583 (95.83%)
  Questions with Full Agreement: 92/96
  Disagreement Rate: 0.0417 (4.17%)