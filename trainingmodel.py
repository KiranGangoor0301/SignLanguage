from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Creates a dictionary that maps action labels to numerical values using a dictionary comprehension. actions might represent different hand gestures or actions.
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
# Initializes empty lists to hold sequences of hand gesture data (sequences) and their corresponding labels (labels).   
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Loads the keypoints data (numpy arrays) for each frame of a sequence. 
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            #  Adds the loaded keypoints data (res) for each frame to the window list.
            window.append(res)
            # Appends the window (list of frames' keypoints) into the sequences list.
        sequences.append(window)
        #  Appends the corresponding label (mapped from action to a numerical value) into the labels list for each sequence.
        labels.append(label_map[action])

# Converts the list of sequences (sequences) into a numpy array (X).
X = np.array(sequences)
y = to_categorical(labels).astype(int)
# Converts the categorical labels (labels) into a one-hot encoded format using to_categorical from Keras.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# Specifies that 5% of the data will be used for testing, while the remaining 95% will be used for training.

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')




# Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to overcome the limitations of traditional RNNs in capturing and learning from long-term dependencies or sequences in data.

# Memory Cell (Long-Term Memory):

# LSTM units (or cells) contain a memory cell that can maintain information over long sequences.
# This memory cell enables LSTMs to remember and retain information for extended periods, preventing the "vanishing gradient" problem encountered by traditional RNNs.

# LSTMs have three gates:
# Forget Gate: Determines what information from the cell state to forget or discard.
# Input Gate: Decides which new information to store in the cell state.
# Output Gate: Controls what information gets output based on the current input and the memory.

# Information Flow:

# LSTMs use these gates to regulate the flow of information, deciding what to remember, forget, and update over time.
# This ability to selectively retain and discard information makes LSTMs well-suited for learning and remembering patterns in sequential data.

# LSTMs address the challenge of capturing long-term dependencies in sequential data by allowing neural networks to remember information over extended sequences.
# They enable the network to retain relevant information and discard unnecessary details, facilitating better learning and understanding of patterns within sequences.


# LSTM Layers:

# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63))):
# LSTM layer with 64 units, returning sequences, ReLU activation, and expecting an input shape of (30, 63).
# Assumes input sequences of 30 time steps (frames) with 63 features (likely keypoints or attributes per frame).
# model.add(LSTM(128, return_sequences=True, activation='relu')):
# Another LSTM layer with 128 units, returning sequences with ReLU activation.
# model.add(LSTM(64, return_sequences=False, activation='relu')):
# Third LSTM layer with 64 units, not returning sequences, using ReLU activation.
# Typically used to consolidate information from the sequence into a fixed-sized representation.
# Dense Layers:

# model.add(Dense(64, activation='relu')):
# Dense layer with 64 units and ReLU activation.
# model.add(Dense(32, activation='relu')):
# Another Dense layer with 32 units and ReLU activation.
# Output Layer:

# model.add(Dense(actions.shape[0], activation='softmax')):
# Final Dense layer with units equal to the number of unique actions (actions.shape[0]), using softmax activation.
# Intended for multi-class classification, generating probabilities for different action classes.
# Variable res (Not Used):

# Variable res is defined but not utilized within this model architecture setup. It seems to be a list with three elements ([.7, 0.2, 0.1]), but it doesn't have any direct impact on the model definition or training.

# This architecture aims to process sequences of data, likely representing hand gestures or actions, using LSTM layers to capture temporal dependencies and eventually classify these sequences into different action categories using the provided sequential data. The final softmax layer provides the probability distribution over the action classes.



# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']):
# Optimizer: The model's optimizer is set to Adam, a popular optimization algorithm used for training neural networks.
# Loss Function: Categorical Cross-Entropy is selected as the loss function, suitable for multi-class classification tasks.
# Metrics: The model will evaluate its performance during training using categorical accuracy as the metric, which measures the proportion of correctly predicted classes among all predictions made.
# Model Training:
# model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback]):
# Training Data: X_train and y_train are the input sequences and their corresponding labels, respectively, used for training the model.
# Epochs: The model will be trained for 200 epochs, iterating over the training data for this number of times.
# Callbacks: The TensorBoard callback tb_callback is provided to enable tracking and visualization of training metrics using TensorBoard.

# model.summary(): Generates a summary of the model architecture, displaying details such as the types and shapes of layers, the number of parameters, and the model's structure.