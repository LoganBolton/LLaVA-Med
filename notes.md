
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

## Condensed Results - Crop

### CT Scan

| Crop Level | LLava-Med-1.5-7B Accuracy | MedGemma4B Accuracy |
|------------|----------------------------|-------------------|
| 0.91       | 0.2939                     | 0.3387            |
| 0.93       | 0.2985                     | 0.3617            |
| 0.95       | 0.2997                     | 0.3387            |
| 0.97       | 0.2997                     | 0.3536            |
| 0.99       | 0.2928                     | 0.3639            |
| 1.0        | 0.2939                     | 0.3536            |

---

### Covid-19 Tianchi

| Crop Level | LLava-Med-1.5-7B Accuracy | MedGemma4B Accuracy |
|------------|----------------------------|-------------------|
| 0.91       | 0.4062                     | 0.5729            |
| 0.93       | 0.4167                     | 0.5521            |
| 0.95       | 0.4062                     | 0.6562            |
| 0.97       | 0.4062                     | 0.5833            |
| 0.99       | 0.4271                     | 0.6042            |
| 1.0        | 0.4167                     | 0.6250            |

### Agreement Rates

| Dataset          | LLava-Med-1.5-7B Agreement Rate | MedGemma Agreement Rate |
|------------------|---------------------------------|--------------------------|
| CT Scan          | 92.77%                 | 48.34%          |
| Covid-19 Tianchi |95.83%                 | 68.75%          |


## LLava-Med-1.5-7B

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


## MedGemma

### CT Scan
| crop Level | Accuracy | Correct | Total |
|------------|----------|---------|-------|
| 0.91       | 0.3387   | 295     | 871   |
| 0.93       | 0.3617   | 315     | 871   |
| 0.95       | 0.3387   | 295     | 871   |
| 0.97       | 0.3536   | 308     | 871   |
| 0.99       | 0.3639   | 317     | 871   |
| 1.0        | 0.3536   | 308     | 871   |

Agreement Analysis:
  Overall Agreement Rate: 0.4834 (48.34%)
  Questions with Full Agreement: 421/871
  Disagreement Rate: 0.5166 (51.66%)

### Covid tianchi

| crop Level | Accuracy | Correct | Total |
|------------|----------|---------|-------|
| 0.91       | 0.5729   | 55      | 96    |
| 0.93       | 0.5521   | 53      | 96    |
| 0.95       | 0.6562   | 63      | 96    |
| 0.97       | 0.5833   | 56      | 96    |
| 0.99       | 0.6042   | 58      | 96    |
| 1.0        | 0.6250   | 60      | 96    |

Agreement Analysis:
  Overall Agreement Rate: 0.6875 (68.75%)
  Questions with Full Agreement: 66/96
  Disagreement Rate: 0.3125 (31.25%)


## Contrast

### MedGemma

Chest CT Scan

| contrast Level | Accuracy | Correct | Total |
|------------|----------|---------|-------|
| 1.0        | 0.3536   | 308     | 871   |
| 1.05       | 0.3605   | 314     | 871   |
| 1.1        | 0.3536   | 308     | 871   |
| 1.15       | 0.3571   | 311     | 871   |
| 1.2        | 0.3594   | 313     | 871   |
| 1.25       | 0.3617   | 315     | 871   |

Agreement Analysis:
  Overall Agreement Rate: 0.5786 (57.86%)
  Questions with Full Agreement: 504/871
  Disagreement Rate: 0.4214 (42.14%)

