import os

import joblib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import streamlit as st

import torch

from pytube.exceptions import RegexMatchError

from torchaudio import load
from torchaudio.transforms import Resample

from src import utils

st.set_page_config(
    page_title="NAML - Music Genre Classifcation 🎼",
    page_icon="🎼"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.set_cmap("Greys")  # Set cmap for spectrogram
color_palette = sns.color_palette("Set3", n_colors=10)

genre_dict = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "Hiphop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

SAMPLE_RATE = 22_050
MODEL_DIR = "models/SENet-GAP-w-Data-Aug"

# --- Load the necessary components to represent the model's latent space ---
training_embeddings = st.cache_data(joblib.load)(os.path.join(MODEL_DIR, "training_embeddings.pkl"))
pca = st.cache_data(joblib.load)(os.path.join(MODEL_DIR, "PCA.pkl"))
scaler = st.cache_data(joblib.load)(os.path.join(MODEL_DIR, "StdScaler.pkl"))
trues = st.cache_data(joblib.load)(os.path.join(MODEL_DIR, "training_labels.pkl"))

with st.sidebar:
    st.session_state["max_duration"] = st.number_input("Max duration in seconds", value=60, step=10, max_value=600)
    st.info("Maximum duration (in seconds) that the model will process.")
    st.session_state["slice_duration"] = st.number_input("Duration of slices", value=2.0, step=0.5, max_value=10.0)
    st.info("Duration of each individual slice extracted from the audio.")
    st.session_state["overlap"] = st.slider("Overlap", value=0.5, step=0.05, max_value=1.0)
    st.info("The fraction of overlap between each contigous slice.")


# --- PAGE START ---

st.image("images/logo_polimi.png")
st.divider()

# ------------------

st.subheader("NAML 2022: Practical project")
st.title("Automatic music genre classification with Deep Learning")

# ------------------

st.divider()
st.markdown("")

left, right = st.columns((1, 3))
with left:
    st.markdown("#### Upload a song extract for our model to classify")

wav = None
thumbnail_url = None
with right:
    audio_file = st.file_uploader("uploader", label_visibility="collapsed")
    url = st.text_input("Or paste a YouTube URL:")
    if audio_file:
        # Load a file using torchaudio backend
        # will only support specific file formats
        try:
            song_name = audio_file.name
            wav, sr = load(audio_file)
            wav = Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
            print("wav_mean:", torch.mean(wav))
        except RuntimeError:
            st.error("There was an issue loading the file. The file format might not be supported by torchaudio"
                     "backend. Please refer to"
                     "[torchaudio backend documentation](https://pytorch.org/audio/stable/backend.html).")
    elif url:
        try:
            audio_io, song_name = utils.get_audio_stream_from_youtube(url)
            wav, sr = load(audio_io)
        except RegexMatchError:
            st.error("Could not resolve the specified URL.")
        except Exception as err:
            st.error(f"**ERROR**: Encountered `{err.__class__.__name__}`")
            raise err.with_traceback(err.__traceback__)

# ------ IF AUDIO -------

if wav is not None:
    st.divider()
    st.markdown("")
    st.write('Song name:', song_name)

    # --- Load pre-trained model and its associated transform (power-spectrogram) ---
    model, transform = utils.load_model_and_transform(MODEL_DIR, checkpoint="accuracy", model_type="CNN")
    model.eval()  # Set the model to inference mode

    wav = wav[:1, :]  # Make audio mono by dropping all subsequent channels

    # --- Show audio component for playback ----
    st.audio(wav.numpy(), sample_rate=SAMPLE_RATE)

    # --- Show a random ~10-sec section of the associated spectrogram ---
    spec = transform(wav).squeeze().numpy()
    offset = 0 if spec.shape[-1] <= 1200 else np.random.randint(spec.shape[-1] - 1200)
    spec = np.log1p(spec)[400::-1, offset:offset+1200]
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.axis(False)
    ax.imshow(spec)
    st.pyplot(fig)

    # ------------------

    st.divider()
    left, right = st.columns(2)
    left.header("Inference")

    # --- Slice up the audio into 4.0 sec extracts ---
    slices = utils.slice_audio(
        wav=wav,
        slice_duration=st.session_state["slice_duration"],
        overlap=st.session_state["overlap"],
        sample_rate=SAMPLE_RATE,
        max_duration=st.session_state["max_duration"]
    )
    print("NUM SLICES:", len(slices))

    # --- Show the color palette associated with the genres ---
    palplot = sns.palplot(color_palette)
    plt.axis(False)
    for k, genre in enumerate(genre_dict):
        plt.text(k, 0.05, genre_dict[k],
                 horizontalalignment="center",
                 fontdict={
                     "family": "sans-serif",
                     "weight": "semibold",
                     "size": 10,
                 })
    st.pyplot(palplot)

    # -----INFERENCE-----

    figure = st.empty()  # Empty component for the predictions stacked chart
    container = st.container()  # Container for the feature-space figure and top-5 genres

    metrics_column, feature_column = container.columns((1, 4))  # Split container in 2 parts
    feature_figure = feature_column.empty()  # Initialize empty component for feature space figure
    metrics = [metrics_column.empty() for i in range(5)]  # Initialize empty components for top-5 genre probabilities

    # --- Make the figure representing the training set feature space representation ---
    proj = pca.transform(scaler.transform(training_embeddings))
    trues = [genre_dict[int(k)] for k in trues]
    training_feat_space = px.scatter_3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        color=trues,
        color_discrete_map={
            k: v for k, v in zip(genre_dict.values(), px.colors.qualitative.Set3)
        }  # Make the genres properly match the colors used above
    )
    training_feat_space.update_traces(marker_size=5)
    feature_figure.plotly_chart(training_feat_space, use_container_width=True)

    # --- Initialize figure for predictions stackplot ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_prop_cycle('color', color_palette)
    fig.tight_layout()
    plt.axis(False)
    ax.set_xlim(0, len(slices)-1)

    # --- Initialize registration of feature vector at forward pass ---
    activation = {}
    model.pool.register_forward_hook(utils.get_activation("embedding", activation))

    total_probas = torch.Tensor()
    total_feats = torch.Tensor()

    with torch.no_grad():
        for i, x in enumerate(slices):
            x = x.unsqueeze(0)  # Add dummy batch dimension
            x = transform(x)  # Compute spectrogram
            probas = model(x)  # Compute model's forward pass

            # Amplify the probas without resorting to softmax in order
            # not to output overly confident predictions
            probas -= torch.min(probas)
            probas **= 10
            probas /= torch.sum(probas)

            # probas = torch.nn.functional.softmax(probas, dim=1)
            feats = activation["embedding"]

            total_feats = torch.concatenate((total_feats, feats), dim=0)
            total_probas = torch.concatenate((total_probas, probas), dim=0)

            mean_probas = torch.mean(total_probas, dim=0).numpy() * 100
            top_classes = np.argsort(mean_probas)[::-1]

            # --- Update the stackplot of genre probabilities ---
            ax.stackplot(torch.arange(i+1), *torch.vsplit(total_probas.T, 10))
            figure.pyplot(fig)

            # --- Update top-5 genres probabilities metrics ---
            for k, m in enumerate(metrics):
                m.metric(f"{genre_dict[top_classes[k]]}", f"{mean_probas[top_classes[k]]:5>.2f} %")

    # --- Project the feature vector into the 3D PCA representation ---
    pca_proj_feats = pca.transform(scaler.transform(total_feats.squeeze()))
    centroid = np.mean(pca_proj_feats, axis=0)
    sample_feat_space = px.line_3d(
        x=pca_proj_feats[:, 0], y=pca_proj_feats[:, 1], z=pca_proj_feats[:, 2],
        markers=True,
    )
    centroid = px.scatter_3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]])
    centroid.update_traces(marker_size=15, marker_line_width=3, marker_line_color="red", marker_color="black")
    sample_feat_space.update_traces(line_color='#000000', line_width=3, marker_size=7)
    feature_space = go.Figure(data=training_feat_space.data + sample_feat_space.data + centroid.data)
    feature_figure.plotly_chart(feature_space, use_container_width=True)
