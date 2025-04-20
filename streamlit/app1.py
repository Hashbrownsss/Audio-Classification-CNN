# app.py
import streamlit as st
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import img_to_array, load_img # Added for inference preprocessing
import pathlib
import time  # To measure time
import io # To handle uploaded file in memory

# --- Configuration & Constants ---
# Using raw strings for Windows paths is correct
DEFAULT_AUDIO_DIR = r"C:\Users\HARSH\Desktop\cnn\Voice of Birds"
DEFAULT_SPECTROGRAM_DIR = r"C:\Users\HARSH\Desktop\cnn\spec_images"

# --- Helper Functions (Spectrogram Generation - Keep As Is) ---

@st.cache_data(show_spinner=False) # Cache spectrogram generation results for a given file
def generate_and_save_spectrogram_image(audio_path, output_image_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512, fmax=8000, fixed_duration=2.0):
    """Generates and saves a single spectrogram image."""
    try:
        # Load audio with fixed duration
        y, current_sr = librosa.load(audio_path, sr=sr, duration=fixed_duration)

        # Pad if shorter than fixed duration
        if len(y) < fixed_duration * sr:
            y = np.pad(y, (0, int(fixed_duration * sr) - len(y)), mode='constant')
        # Truncate if longer (shouldn't happen with duration=fixed_duration, but safety)
        elif len(y) > fixed_duration * sr:
             y = y[:int(fixed_duration * sr)]

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot and save
        fig = plt.figure(figsize=(4, 4)) # Use fixed size matching IMG_WIDTH/HEIGHT ratio if possible
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, fmax=fmax, cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=(IMG_WIDTH // 4)) # Adjust dpi to match target size approx
        plt.close(fig) # Close the specific figure

        return output_image_path, None # Return path on success, None for error message

    except Exception as e:
        plt.close('all') # Close any potentially open figures on error
        return None, f"Error processing {os.path.basename(audio_path)}: {e}"

# --- Inference Helper Function ---
def generate_spectrogram_for_inference(audio_data, sr=22050, n_mels=128, n_fft=2048, hop_length=512, fmax=8000, fixed_duration=2.0):
    """Generates a spectrogram matrix from audio data (bytes or path) for inference."""
    try:
        y, current_sr = librosa.load(audio_data, sr=sr, duration=fixed_duration)
        if len(y) < fixed_duration * sr:
            y = np.pad(y, (0, int(fixed_duration * sr) - len(y)), mode='constant')
        elif len(y) > fixed_duration * sr:
            y = y[:int(fixed_duration * sr)]

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        st.error(f"Error processing audio for inference: {e}")
        return None
sr = 22050
hop_length =512
def preprocess_spectrogram_for_model(spec_db, img_height, img_width):
    """Converts spectrogram matrix to model-ready image tensor."""
    # Plot spectrogram to a buffer to simulate image loading
    fig = plt.figure(figsize=(img_width/100, img_height/100)) # Size in inches based on DPI=100
    librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, fmax=8000, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    # Load image from buffer and resize
    img = tf.keras.utils.load_img(buf, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # The model's Rescaling layer handles the 1./255
    return img_array


# --- Training Helper Functions (Keep As Is) ---
def run_spectrogram_generation(dataset_path, output_dir, sr, n_mels, n_fft, hop_length, fixed_duration):
    # (Your existing code for run_spectrogram_generation)
    # ... (ensure it returns image_paths, labels)
    st.info(f"Starting spectrogram generation from: {dataset_path}")
    st.write(f"Outputting images to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    labels = []
    processed_count = 0
    error_count = 0
    errors_list = []
    start_time = time.time()

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    # Estimate total files carefully
    total_files_estimate = 0
    try:
        for _, _, files in os.walk(dataset_path):
            total_files_estimate += len([f for f in files if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))])
    except Exception as e:
        st.warning(f"Could not accurately estimate total files: {e}")
        total_files_estimate = 1 # Avoid division by zero

    files_processed_so_far = 0

    for root, dirs, files in os.walk(dataset_path):
        class_label = os.path.basename(root)
        if root == dataset_path:
             if not dirs and any(f.lower().endswith((".mp3", ".wav", ".flac", ".ogg")) for f in files):
                 st.warning(f"Audio files found directly in root '{dataset_path}'. Processing them under 'unknown_class'.")
                 class_label = "unknown_class"
                 class_output_dir = os.path.join(output_dir, class_label)
                 os.makedirs(class_output_dir, exist_ok=True)
             else:
                continue

        else: # Only process subdirectories if they exist
            class_output_dir = os.path.join(output_dir, class_label)
            os.makedirs(class_output_dir, exist_ok=True)

        status_placeholder.text(f"Processing class: {class_label}...")
        audio_files = [f for f in files if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]
        if not audio_files:
            continue

        for file in audio_files:
            files_processed_so_far += 1
            audio_path = os.path.join(root, file)
            image_filename = os.path.splitext(file)[0] + ".png"
            output_image_path = os.path.join(class_output_dir, image_filename)

            saved_path, error_msg = generate_and_save_spectrogram_image(
                audio_path, output_image_path, sr, n_mels, n_fft, hop_length, fixed_duration=fixed_duration
            )

            if saved_path:
                image_paths.append(saved_path)
                labels.append(class_label)
                processed_count += 1
            else:
                error_count += 1
                if error_msg:
                     errors_list.append(error_msg)

            if total_files_estimate > 0:
                progress_bar.progress(min(1.0, files_processed_so_far / total_files_estimate))
            else:
                # Fallback if estimate failed
                 progress_bar.progress(0.5) # Indicate some progress


    end_time = time.time()
    progress_bar.progress(1.0)
    status_placeholder.empty()

    st.success(f"Spectrogram generation finished in {end_time - start_time:.2f} seconds.")
    st.write(f"Successfully generated: {processed_count} images.")
    st.write(f"Encountered errors: {error_count} files.")
    if errors_list:
         with st.expander("Show Errors"):
              for err in errors_list:
                   st.warning(err)

    if processed_count == 0:
        st.error("No spectrogram images were generated. Check dataset path, file formats, and permissions.")
        return None, None

    return np.array(image_paths), np.array(labels)


def plot_one_image_per_class(image_paths, labels, max_classes_to_plot=15):
    # (Your existing code for plot_one_image_per_class)
    # ...
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    if num_classes == 0:
        st.warning("No classes found in generated labels to plot.")
        return

    st.subheader("Example Spectrogram Images per Class")
    num_to_plot = min(num_classes, max_classes_to_plot)
    if num_classes > max_classes_to_plot:
        st.write(f"Displaying examples for the first {max_classes_to_plot} classes found.")

    fig_height = max(4, 2 * num_to_plot)
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(6, fig_height))
    if num_to_plot == 1:
        axes = [axes]

    for i, class_label in enumerate(unique_classes[:num_to_plot]):
        try:
             index = np.where(labels == class_label)[0][0]
             img_path = image_paths[index]
             ax = axes[i]
             img = plt.imread(img_path)
             ax.imshow(img)
             ax.set_title(f"Class: {class_label}")
             ax.axis('off')
        except IndexError:
             st.warning(f"Could not find image for class: {class_label}")
             axes[i].set_title(f"Class: {class_label} (Image not found)")
             axes[i].axis('off')
        except Exception as e:
            st.error(f"Error loading image for {class_label}: {e}")
            axes[i].set_title(f"Class: {class_label} (Error loading image)")
            axes[i].axis('off')


    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --- Initialize Session State ---
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🐦 Bird Voice Spectrogram Classification")

st.sidebar.header("Configuration")

# --- Sidebar Inputs ---
data_path = st.sidebar.text_input("Audio Dataset Root Directory", DEFAULT_AUDIO_DIR)
spec_out_path = st.sidebar.text_input("Output Spectrogram Directory", DEFAULT_SPECTROGRAM_DIR)

st.sidebar.subheader("Librosa Parameters")
# Make these consistent between training and inference
sr_param = st.sidebar.number_input("Sample Rate (sr)", value=22050, step=100)
n_mels_param = st.sidebar.number_input("Number of Mel bins (n_mels)", value=128, min_value=32, max_value=512, step=16)
fixed_duration_param = st.sidebar.slider("Fixed Duration (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
n_fft_param = st.sidebar.number_input("FFT Window Size (n_fft)", value=2048, min_value=256, max_value=8192, step=256)
hop_length_param = st.sidebar.number_input("Hop Length", value=512, min_value=64, max_value=2048, step=64)

st.sidebar.subheader("Training Parameters")
BATCH_SIZE = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)
EPOCHS = st.sidebar.slider("Max Epochs (Early Stopping Enabled)", 10, 200, 30, step=5)
VALIDATION_SPLIT = st.sidebar.slider("Validation Split", 0.1, 0.5, 0.2, step=0.05)
SEED = st.sidebar.number_input("Random Seed", value=42)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Button to start the process
start_button = st.sidebar.button("Generate Spectrograms & Train Model", type="primary")

# --- Main Area ---

# Section 1 & 2: Spectrogram Generation and Training (only runs if button is clicked)
if start_button:
    st.session_state.model_trained = False # Reset status on new training run
    st.session_state.trained_model = None
    st.session_state.class_names = None

    st.header("1. Spectrogram Generation")
    image_dir_path = pathlib.Path(spec_out_path)

    with st.spinner("Generating spectrogram images... This may take a while."):
        generated_image_paths, generated_labels = run_spectrogram_generation(
            data_path, spec_out_path, sr_param, n_mels_param, n_fft_param, hop_length_param, fixed_duration_param
        )

    if generated_image_paths is not None and len(generated_image_paths) > 0:
        plot_one_image_per_class(generated_image_paths, generated_labels)
        st.header("2. Model Training")
        st.subheader("Loading Datasets")
        try:
            # --- Load Data ---
            with st.spinner("Loading training dataset..."):
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    image_dir_path, validation_split=VALIDATION_SPLIT, subset="training", seed=SEED,
                    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode='categorical'
                )
            train_files_count = train_ds.cardinality().numpy() * BATCH_SIZE if train_ds.cardinality().numpy() > 0 else len(list(image_dir_path.rglob('*/*.png'))) * (1-VALIDATION_SPLIT) # Estimate
            st.write(f"Found approximately {int(train_files_count)} training files.")

            with st.spinner("Loading validation dataset..."):
                 val_ds = tf.keras.utils.image_dataset_from_directory(
                    image_dir_path, validation_split=VALIDATION_SPLIT, subset="validation", seed=SEED,
                    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode='categorical'
                )
            val_files_count = val_ds.cardinality().numpy() * BATCH_SIZE if val_ds.cardinality().numpy() > 0 else len(list(image_dir_path.rglob('*/*.png'))) * (VALIDATION_SPLIT) # Estimate
            st.write(f"Found approximately {int(val_files_count)} validation files.")

            if train_ds.cardinality().numpy() == 0 or val_ds.cardinality().numpy() == 0:
                 st.error(f"Failed to load data from {image_dir_path}.")
                 st.stop()

            class_names_loaded = train_ds.class_names
            num_classes = len(class_names_loaded)
            st.write(f"**Classes found:** {num_classes} -> {', '.join(class_names_loaded)}")

            # --- Optimize Datasets ---
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            st.write("Applied dataset caching, shuffling (train), and prefetching.")

            # --- Build Model ---
            st.subheader("Building CNN Model")
            model = Sequential([
                layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                layers.Conv2D(32, (3, 3), padding='same', activation='relu'), layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), padding='same', activation='relu'), layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding='same', activation='relu'), layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'), layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ], name="Original_Bird_CNN")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            st.subheader("Model Summary")
            with st.expander("Show Summary"): model.summary(print_fn=lambda x: st.text(x))

            # --- Callbacks ---
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
            st.write("Configured Early Stopping (patience=10) and ReduceLROnPlateau (patience=5).")

            # --- Train Model ---
            st.subheader("Training Progress")
            st.info(f"Starting training for up to {EPOCHS} epochs...")
            history = None
            training_placeholder = st.empty()
            with st.spinner("Training in progress... See console/terminal for epoch details."):
                start_train_time = time.time()
                history = model.fit(
                    train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=[early_stopping, reduce_lr], verbose=0
                )
                end_train_time = time.time()

            if history:
                training_placeholder.success(f"Training finished in {end_train_time - start_train_time:.2f} seconds!")
                actual_epochs = len(history.history['loss'])
                st.write(f"Training stopped after {actual_epochs} epochs.")

                # --- Evaluate ---
                st.subheader("Evaluation Results (Best Weights)")
                with st.spinner("Evaluating model..."): loss, accuracy = model.evaluate(val_ds, verbose=0)
                col1, col2 = st.columns(2)
                col1.metric("Validation Loss", f"{loss:.4f}")
                col2.metric("Validation Accuracy", f"{accuracy:.4f}")

                # --- Plot History ---
                st.subheader("Training History")
                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                loss_hist = history.history['loss']
                val_loss_hist = history.history['val_loss']
                epochs_range = range(actual_epochs)
                fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                best_epoch = np.argmin(val_loss_hist)
                # Plot Accuracy
                ax1.plot(epochs_range, acc, label='Training Accuracy'); ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
                ax1.scatter(best_epoch, val_acc[best_epoch], color='red', label=f'Best Val Loss Epoch ({best_epoch+1})', zorder=5)
                ax1.legend(loc='lower right'); ax1.set_title('Training and Validation Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.grid(True)
                # Plot Loss
                ax2.plot(epochs_range, loss_hist, label='Training Loss'); ax2.plot(epochs_range, val_loss_hist, label='Validation Loss')
                ax2.scatter(best_epoch, val_loss_hist[best_epoch], color='red', label=f'Best Val Loss Epoch ({best_epoch+1})', zorder=5)
                ax2.legend(loc='upper right'); ax2.set_title('Training and Validation Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.grid(True)
                st.pyplot(fig_hist); plt.close(fig_hist)

                # --- Store Model and Class Names in Session State ---
                st.session_state.model_trained = True
                st.session_state.trained_model = model
                st.session_state.class_names = class_names_loaded
                st.success("Model trained successfully and is ready for prediction.")

            else: st.error("Model training failed.")
        except FileNotFoundError: st.error(f"Error: Input spectrogram directory not found at '{image_dir_path}'.")
        except Exception as e: st.error(f"An error occurred during training: {e}"); st.exception(e)
    elif start_button and generated_image_paths is None:
         st.warning("Spectrogram generation failed. Cannot proceed to training.")

# Section 3: Inference (only shows if model is trained in the current session)
st.divider() # Add a visual separator
st.header("3. Predict Bird Sound")

if st.session_state.model_trained and st.session_state.trained_model is not None and st.session_state.class_names is not None:
    uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .ogg, .flac)", type=['wav', 'mp3', 'ogg', 'flac'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)

        # Use parameters defined in the sidebar for consistency
        inference_sr = sr_param
        inference_n_mels = n_mels_param
        inference_n_fft = n_fft_param
        inference_hop_length = hop_length_param
        inference_duration = fixed_duration_param

        # Process the uploaded file
        with st.spinner("Processing audio and generating spectrogram..."):
            # Librosa needs a file path or an object with 'read' method.
            # For uploaded files, pass the file object directly.
            spec_db = generate_spectrogram_for_inference(
                uploaded_file,
                sr=inference_sr, n_mels=inference_n_mels, n_fft=inference_n_fft,
                hop_length=inference_hop_length, fixed_duration=inference_duration
            )

        if spec_db is not None:
            st.subheader("Generated Spectrogram for Prediction")
            fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
            # --- This line causes the error in the original code ---
            # librosa.display.specshow(spec_db, sr=inference_sr, hop_length=inference_hop_length, x_axis='time', y_axis='mel', ax=ax_spec, cmap='magma')
            # fig_spec.colorbar(ax_spec.pcolormesh(np.arange(spec_db.shape[1]+1)*inference_hop_length/inference_sr, librosa.mel_frequencies(n_mels=inference_n_mels), spec_db, cmap='magma'), ax=ax_spec, format='%+2.0f dB')
            # --- End of problematic block ---

            # === Replace with this corrected block: ===
            # 1. Call specshow and capture the returned object (often a QuadMesh)
            img = librosa.display.specshow(spec_db,
                                           sr=inference_sr,
                                           hop_length=inference_hop_length,
                                           x_axis='time',
                                           y_axis='mel',
                                           fmax=8000,  # Match fmax if used elsewhere, e.g., generation
                                           ax=ax_spec,
                                           cmap='magma')

            # 2. Pass the captured 'img' object to colorbar
            fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
            # === End of corrected block ===

            st.pyplot(fig_spec)
            plt.close(fig_spec)

            # Preprocess for model (This part should be fine)
            with st.spinner("Preparing spectrogram for model..."):
                img_array = preprocess_spectrogram_for_model(spec_db, IMG_HEIGHT, IMG_WIDTH)

            # Predict
            with st.spinner("Predicting..."):
                model_to_predict = st.session_state.trained_model # Load from session state
                predictions = model_to_predict.predict(img_array)
                score = tf.nn.softmax(predictions[0]) # Apply softmax if model output isn't already probabilities

            predicted_class_index = np.argmax(score)
            predicted_class_name = st.session_state.class_names[predicted_class_index]
            confidence = 100 * np.max(score)

            st.subheader("Prediction")
            st.success(f"Predicted Bird: **{predicted_class_name}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

            # Optional: Show top N predictions
            with st.expander("Show Top Predictions"):
                top_indices = np.argsort(score)[::-1][:5] # Get top 5 indices
                for i in top_indices:
                     st.write(f"- {st.session_state.class_names[i]}: {100 * score[i]:.2f}%")

        else:
            st.error("Could not generate spectrogram from the uploaded file.")

else:
    st.info("Train a model first (using the button in the sidebar) to enable prediction.")

# Add footer or other info if needed
st.sidebar.divider()
st.sidebar.info("App by HARSH using Librosa, TensorFlow/Keras, and Streamlit.")