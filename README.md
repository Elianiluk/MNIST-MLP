
# **Handwritten Digit Classification Using a Fully Connected Neural Network**

## **Overview**
This program implements a simple fully connected neural network to classify handwritten digits from the **MNIST** dataset. The dataset consists of grayscale images of digits (0-9), and the model is trained to predict the correct digit for a given input image.

### **Key Features**
1. **Fully Connected Neural Network**:
   - 4 Fully Connected Layers.
   - ReLU activation functions.
   - Dropout layers for regularization.
   - Softmax for output probabilities.

2. **Dataset**:
   - **MNIST** dataset containing 60,000 training images and 10,000 test images.

3. **Training and Evaluation**:
   - Trains the network using the Negative Log Likelihood Loss (`NLLLoss`).
   - Uses the Adam optimizer for efficient weight updates.

4. **Visualization**:
   - Displays training images and corresponding labels.
   - Displays test images with predicted and true labels, color-coded for correctness.

---

## **Usage**

### **Requirements**
Install the required libraries before running the code:
```bash
pip install torch torchvision matplotlib numpy
```

### **File Structure**
- `mnist_nn.py`: The main script containing the implementation.
- `data/`: Directory where the MNIST dataset will be downloaded.

### **Run the Program**
1. Clone the repository or download the script.
2. Run the program:
   ```bash
   python mnist_nn.py
   ```

---

## **Program Workflow**
1. **Data Loading and Preprocessing**:
   - Loads the MNIST dataset using `torchvision.datasets`.
   - Converts images to PyTorch tensors.

2. **Model Definition**:
   - Implements a fully connected neural network with 4 layers.
   - Includes dropout for regularization.

3. **Training**:
   - Trains the model for a specified number of epochs (default: 30).
   - Monitors training loss at the end of each epoch.

4. **Testing**:
   - Evaluates the model on the test set.
   - Computes per-class and overall accuracy.

5. **Visualization**:
   - Displays training images with their labels.
   - Displays test images with predicted and true labels.

---

## **Model Architecture**
| Layer Type            | Parameters                      |
|-----------------------|---------------------------------|
| Fully Connected Layer | Input: 784 (28x28), Output: 256 |
| Fully Connected Layer | Input: 256, Output: 128         |
| Fully Connected Layer | Input: 128, Output: 64          |
| Fully Connected Layer | Input: 64, Output: 10 (classes) |
| Activation Function    | ReLU                           |
| Dropout               | Probability: 0.2               |
| Output Function       | LogSoftmax                     |

---

## **Customization**
1. **Adjust Training Parameters**:
   - Modify the number of epochs, batch size, or learning rate as needed:
     ```python
     n_epochs = 50
     batch_size = 32
     learning_rate = 0.001
     ```

2. **Dataset**:
   - Replace MNIST with a different dataset by modifying the data loaders.

3. **Model Architecture**:
   - Experiment with different numbers of layers or neurons to improve performance.

---

## **Results**
1. **Training Loss**:
   - Displays the loss at the end of each epoch.

2. **Test Accuracy**:
   - Per-class accuracy for all 10 digits (0-9).
   - Overall accuracy percentage.

3. **Visual Predictions**:
   - Displays test images with predicted and actual labels:
     - **Green**: Correct predictions.
     - **Red**: Incorrect predictions.

---

## **Contact**
**Author**: Elian Iluk  
**Email**: elian10119@gmail.com  

Feel free to reach out for any questions or feedback regarding the program.

