# Melody-generation-with-lstm-neural-networks

A deep learning project that leverages Long Short-Term Memory (LSTM) neural networks to generate new audio melodies based on existing audio files. By learning the sequence patterns of Mel-frequency cepstral coefficients (MFCCs) from major and minor audio tracks, the model predicts future audio frames and reconstructs them back into listenable `.wav` files.

---

## 🚀 Key Features

* **Audio Feature Extraction:** Utilizes `librosa` to extract MFCCs, reducing complex audio waveforms into trainable numerical representations.
* **LSTM-Based Architecture:** Employs a multi-layer LSTM network to capture the sequential dependencies and temporal patterns of audio sequences.
* **Audio Reconstruction:** Uses mel-to-audio inverse transformations to turn the predicted MFCC arrays back into playable audio.
* **Controlled Variability:** Incorporates optional Gaussian noise during the generation phase to prevent repetitive loops and create unique, natural-sounding melodic variations.
* **Waveform Visualization:** Includes plotting functionality using `matplotlib` to visually compare original and AI-generated audio waveforms.

---

## 🛠️ Prerequisites & Dependencies

To run this notebook, you will need Python 3.x and the following libraries installed:

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib ipython
```

---

## 📂 Dataset Structure

The model expects a specific directory structure for your audio dataset. Create a root folder named `Audio_Files` in the same directory as the notebook, containing two subfolders for major and minor audio tracks (`.wav` format):

```text
├── 22112027_DL_CIA.ipynb
└── Audio_Files/
    ├── major/
    │   ├── Major_1.wav
    │   ├── Major_2.wav
    │   └── ...
    └── minor/
        ├── Minor_1.wav
        ├── Minor_2.wav
        └── ...
```

---

## 🧠 Model Architecture

The neural network is built using TensorFlow/Keras and consists of the following layers:

1. **LSTM Layer:** 64 units, returns sequences.
2. **Dropout Layer:** 20% dropout rate to prevent overfitting.
3. **LSTM Layer:** 64 units.
4. **Dropout Layer:** 20% dropout rate.
5. **Dense Layer:** 32 units with ReLU activation.
6. **Dense Output Layer:** 20 units (matches the number of MFCC features to predict the next frame).

The model is compiled using the `adam` optimizer and evaluates Mean Squared Error (MSE) for loss calculation. It also utilizes Early Stopping to halt training when validation loss stops improving.

---

## ⚙️ How to Use

1. **Clone the Repository:** Download the `.ipynb` file to your local machine.
2. **Prepare the Data:** Set up the `Audio_Files` directory with your own `.wav` files as shown in the Dataset Structure section. 
3. **Run the Notebook:** Execute the cells in order. The notebook will:
   * Load and preprocess the audio files.
   * Create sliding window sequences (length of 30 frames).
   * Train the LSTM model.
   * Save the trained weights as `melody_generator_model.h5`.
   * Evaluate the model using MSE and $R^2$ scores.
4. **Generate Audio:** The final cells take a "seed" `.wav` file (e.g., `Major_8.wav`), extract its initial MFCCs, and prompt the model to generate the next 100 frames. The output can be listened to directly within the Jupyter Notebook interface!

---

## 📊 Evaluation

The model evaluates its performance on unseen test data using two primary metrics:
* **Mean Squared Error (MSE):** Measures the average squared difference between the estimated values and the actual value.
* **$R^2$ Score:** Represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
```

***

### A Quick Tip for GitHub:
Make sure to add `.h5` (your saved model file format) to your `.gitignore` file if the resulting `melody_generator_model.h5` file ends up being larger than 100MB, as GitHub blocks files over that size! 

Let me know if you'd like to adjust the title or add a specific license (like MIT or Apache 2.0) to the file!
