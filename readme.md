# EchoSensAI

**EchoSensAI** is a machine learning project aimed at real-time sound analysis and classification. The project focuses on preprocessing audio datasets from UrbanSound8K, extracting meaningful insights, and building a CRNN machine learning model to classify sounds in real time.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/EchoSensAI.git
   cd EchoSensAI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required Python version (e.g., Python 3.8 or higher).

---

## Usage

### 1. Configure the Project
Rename the `config/config_example.json` file to `config/config.json`
Edit the `config/config.json` file to set the paths and constants for your dataset:
```json
{
    "dataset_path": "path/to/data",
}
```

### 2. Visualize UrbanSound8K Statistics 
Use the provided visualization tools to generate histograms:
```python
from ds_stats import plot_sr_histogram, plot_channel_nb_histogram

# Example usage
plot_sr_histogram(properties_df)
plot_channel_nb_histogram(properties_df)
```
Choose a common sample rate for the audio files

### 3. Run the Preprocessing Pipeline (Coming Soon)
The preprocessing pipeline will be implemented soon. 

### 4. Train the Machine Learning Model (Coming Soon)
The machine learning model for sound classification will be implemented soon. Stay tuned for updates!

---

## Project Structure

```
EchoSensAI/
├── config/                 # Configuration files
│   └── config.json         # JSON configuration file
├── data/                   # Audio dataset (not included in the repo)
├── log/                    # Logs
├── results/                # Results and outputs
├── src/                    # Source code
│   ├── data_loader.py      # Functions for loading and preprocessing audio data
│   ├── utils.py            # Utility functions
├── ds_stats.py             # Functions for computing and visualizing dataset statistics
├── main.py                 # Main script to run the project
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Roadmap

1. **Statistical Analysis**: Compute and visualize dataset statistics (Done).
2. **Preprocessing Pipeline**: Complete the preprocessing of audio datasets (In Progress).
3. **Machine Learning Model**:
   - Implement a CNN for sound classification (In Progress).
   - Extend to a CRNN for improved performance (Future).
4. **Real-Time Sound Classification**: Integrate the model for real-time sound analysis (Future).

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Librosa](https://librosa.org/) for audio processing.
- [Matplotlib](https://matplotlib.org/) for data visualization.
- [Pandas](https://pandas.pydata.org/) for data manipulation.

---

## Contact

For questions or feedback, feel free to open an issue in the repository.
