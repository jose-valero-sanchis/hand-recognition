# Hand Position Classification System

## Project Description

The project consists of a hand position classification system in images. The classified hand positions are: LB (Left Back), LF (Left Front), RB (Right Back) and RF (Right Front). The system uses a pre-trained convolutional neural network (VGG16) to extract features from the images and a dense neural network for classification.

## Image Examples

Here, sample images are provided that showcase different hand positions (LB, LF, RB, RF) and have been utilized to train the model:

<div align="center">
  <table align="center">
    <tr>
      <td align="center"><img src="examples/0147.jpeg" alt="LB"><br><em>LB</em></td>
      <td align="center"><img src="examples/0148.jpeg" alt="LF"><br><em>LF</em></td>
    </tr>
    <tr>
      <td align="center"><img src="examples/0149.jpeg" alt="RB"><br><em>RB</em></td>
      <td align="center"><img src="examples/0150.jpeg" alt="RF"><br><em>RF</em></td>
    </tr>
  </table>
</div>

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/jose-valero-sanchis/hand-recognition.git
    cd hand-recognition
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Train the model and make predictions:

    ```bash
    python main.py <training_directory> <testing_directory>
    ```

## Implementation Details

- **Feature Extraction**: Image features are extracted using the pre-trained [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) network.
- **Model Training**: A dense neural network model is trained using the extracted features and labels from the training images.
- **Testing Image Classification**: Extracted features from testing images are used to make predictions.
