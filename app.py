from flask import Flask, render_template, request
import cv2
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Dense, Flatten, Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model

vocab = np.load('vocab200.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v: k for k, v in vocab.items()}

embedding_size = 128
vocab_size = len(vocab)
max_len = 40

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

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights200.h5')

print("=" * 150)
print("MODEL LOADED")

resnet = load_model('model200.h5')

print("=" * 150)
print("RESNET MODEL LOADED")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab
    
    # Load the uploaded image
    img = request.files['file1']
    img.save('static/file.jpg')
    print("=" * 50)
    print("IMAGE SAVED")
    
    # Load and preprocess the image
    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0  # Normalize image
    image = np.reshape(image, (1, 224, 224, 3))
    print("=" * 50)
    print("Image Processed")
    
    # Extract image features using ResNet
    incept = resnet.predict(image).reshape(1, 2048)
    print("=" * 50)
    print("Image Features Predicted")
    
    # Initialize the initial caption sequence
    text_in = ['startofseq']
    final = ''
    print("=" * 50)
    print("Generating Captions")
    
    # Generate captions iteratively
    count = 0
    while True:
        count += 1
        
        # Tokenize the current sequence and pad it
        encoded = [vocab.get(word, 0) for word in text_in]  # Use 0 for out-of-vocabulary words
        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)
        
        # Predict the next word
        sampled_index = np.argmax(model.predict([incept, padded]))
        sampled_word = inv_vocab.get(sampled_index, '')  # Use empty string for out-of-vocabulary indices
        
        # Check for end-of-sequence token or maximum length
        if sampled_word == 'endofseq' or count >= max_len:
            break
        
        # Append the predicted word to the final caption
        final += ' ' + sampled_word
        
        # Update the input for the next iteration
        text_in.append(sampled_word)
    
    return render_template('after.html', data=final.strip())


if __name__ == "__main__":
    app.run(debug=True)
