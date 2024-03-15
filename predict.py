import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed, Activation, Concatenate
from keras.utils import to_categorical
from keras.applications import ResNet50
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Load vocabulary
vocab = np.load('vocab200.npy', allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}

# Load model architecture
embedding_size = 128
max_len = 40
vocab_size = len(vocab)

# Define model architecture
image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs=out)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Load model weights
model.load_weights('mine_model_weights200.h5')

# Load ResNet model for feature extraction
resnet = load_model('model200.h5')

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0  # Normalize image
    image = np.reshape(image, (1, 224, 224, 3))
    return image

def generate_caption(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess the image using ResNet50's preprocess_input function
    
    # Extract image features using ResNet50
    incept = resnet.predict(img).reshape(1, 2048)

    # Initialize the initial caption sequence
    text_in = ['startofseq']
    final = ''

    # Generate captions iteratively
    count = 0
    while count < 20:  # Limit the number of iterations to prevent infinite loops
        count += 1
        
        # Tokenize the current sequence and pad it
        encoded = [vocab.get(word, 0) for word in text_in]  # Use 0 for out-of-vocabulary words
        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)
        
        # Predict the next word
        sampled_index = np.argmax(model.predict([incept, padded]))
        sampled_word = inv_vocab.get(sampled_index, '')  # Use empty string for out-of-vocabulary indices
        
        # Check for end-of-sequence token
        if sampled_word == 'endofseq':
            break
        
        # Append the predicted word to the final caption
        final += ' ' + sampled_word
        
        # Update the input for the next iteration
        text_in.append(sampled_word)
    
    return final


# Example usage
image_path = 'E:\\Downloads\\Images\\127490019_7c5c08cb11.jpg'
caption = generate_caption(image_path)
print("Generated Caption:", caption)
