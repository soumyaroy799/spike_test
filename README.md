
# spike_test

This repository contains a set of tools to detect, generate, and validate high-frequency spike events in solar X-ray and EUV time series data. These tools are designed for use with solar flare observations and can be applied to datasets from instruments such as STIX, SoLEXS, XSM, or synthetic sources.

## Features

- **Spike Generation** (`spike_gen.py`): Injects synthetic spikes into real or modeled light curves for testing purposes.
- **Spike Detection** (`spike_det.py`): Identifies spike features in time series based on thresholding or morphological characteristics.
- **Spike Verification** (`verify_despike.py`): Cross-checks detected spikes with original data and provides quality control.
- **IDL Support** (`spike_det.pro`): Legacy spike detection implementation in IDL for SoLEXS/XSM workflows.
- **Manual Comparison** (`2spike.py`): Custom routines for visual/manual comparison of spiked and de-spiked profiles.

## Requirements

You can install dependencies using:

```bash
pip install -r requirements.txt
```

##  Usage

### 1. Generate synthetic spikes:
```bash
python spike_gen.py --input lightcurve.csv --output spiked.csv --num-spikes 20
```

### 2. Detect spikes:
```bash
python spike_det.py --input spiked.csv --threshold 5 --window 3
```

### 3. Verify spike removal:
```bash
python verify_despike.py --original spiked.csv --cleaned despiked.csv
```

### 4. Run IDL version (if needed):
```idl
IDL> .run spike_det.pro
```

##  File Structure

```
spike_test/
├── spike_gen.py           # Injects artificial spikes
├── spike_det.py           # Detects spikes from data
├── verify_despike.py      # Compares original and de-spiked curves
├── 2spike.py              # Extra comparison logic
├── spike_det.pro          # IDL-based spike detection
```

## Contact

**Soumya Roy**  
Email: soumyaroy799@gmail.com

##  License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Status

Actively developed and tested on solar datasets from Aditya-L1's SUIT.  
Python 3.8+ recommended.
