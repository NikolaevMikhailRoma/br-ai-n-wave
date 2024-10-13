# Synthetic Seismic Data Generation Project

This project aims to generate synthetic seismic data based on a given oil field using diffusion models. The project consists of several Python scripts that handle different aspects of the data processing, model training, and generation pipeline.

## Project Structure

```
br-ai-n-wave_test/
│
├── data/
│   ├── seismic.sgy
│   ├── normalization_params.pkl
│   └── images/
│       └── generated_seismic.png
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── service.py
│   ├── SeismicSliceReader.py
│   ├── data_preparation.py
│   ├── train_diffusion_model.py
│   └── generate_seismic.py
│
├── tests/
│
├── models/
│   └── diffusion_model.pth
│
├── __pycache__/
│
├── requirements.txt
├── README.md
└── task.md
```

## Key Components

1. **Seismic Data Reading**: The `SeismicSliceReader` class uses the `segfast` library to efficiently read seismic data from SEG-Y files.

2. **Data Preparation**: The `data_preparation.py` script handles data loading, normalization, and preparation for model training.

3. **Diffusion Model**: A UNet-based architecture is implemented in `train_diffusion_model.py` for the diffusion process.

4. **Model Training**: The training loop in `train_diffusion_model.py` handles the diffusion process and model optimization.

5. **Synthetic Data Generation**: `generate_seismic.py` uses the trained model to generate new synthetic seismic data.

## Usage

1. Ensure all required libraries are installed using `pip install -r requirements.txt`.
2. Set the appropriate parameters in `src/config.py`.
3. Run `src/train_diffusion_model.py` to train the model on your seismic data.
4. Use `src/generate_seismic.py` to generate synthetic seismic data with the trained model.

## Current Status

- The basic pipeline for reading seismic data, training a diffusion model, and generating synthetic data is implemented.
- The project uses PyTorch and supports GPU acceleration where available.
- Logging and progress tracking are implemented for better monitoring of the training process.

## Future Improvements

- Implement more sophisticated evaluation metrics for generated seismic data.
- Optimize the model architecture and hyperparameters for better quality synthetic data.
- Add more visualization tools for comparing real and synthetic seismic data.

## Note

This project is still in development. Ensure you have the necessary computational resources, especially for training the diffusion model on large seismic datasets.