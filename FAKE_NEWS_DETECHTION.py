import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# ⬇️ INSERT YOUR FILE PATHS HERE
fake_path = r"C:\Users\ASUS\Desktop\New folder (2)\archive\fake.csv"  # <-- Update this path
true_path = r"C:\Users\ASUS\Desktop\New folder (2)\archive\true.csv"  # <-- Update this path

# Load and label datasets
fake_df = pd.read_csv(fake_path)
fake_df['label'] = 0  # FAKE

true_df = pd.read_csv(true_path)
true_df['label'] = 1  # REAL

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Ensure required columns exist
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

# Keep only needed columns and drop missing values
df = df[['text', 'label']].dropna()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_length = 500
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),  # input_length removed to avoid warning
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=3, validation_data=(X_test_pad, y_test), batch_size=64)

# Predict and evaluate
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
