
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

## Usage

### 1. Detect spikes:
```bash
python spike_det.py --input spiked.csv --threshold 5 --window 3
```

### 2. Verify spike removal:
```bash
python verify_despike.py --original spiked.csv --cleaned despiked.csv
```

### 3. Run IDL version (if you are working in IDL):
```idl
IDL> .run spike_det.pro
```

## File Structure

```
.
├── LICENSE                      # MIT License file
├── README.md                    # Project overview and usage
├── requirements.txt             # Python dependencies
├── output/                      # Output directory
│   └── synthetic_image.fits     # Example FITS output (spiked or processed)
└── src/                         # All source code lives here
    ├── 2spike.py                # Manual comparison logic
    ├── spike_det.pro            # IDL spike detection script
    ├── spike_det.py             # Spike detection in Python
    ├── spike_gen.py             # Spike injection into lightcurve
    └── verify_despike.py        # Despiking verification and visualization
```

## Contact

**Soumya Roy**  
Email: soumyaroy799@gmail.com

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Status

Actively developed and tested on solar datasets from Aditya-L1's SUIT.  
Python 3.8+ recommended.
