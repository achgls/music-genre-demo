import streamlit as st

from pytube.exceptions import RegexMatchError
import joblib

import numpy as np

import torch
from torchaudio import load
from torchaudio.transforms import Resample

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.set_cmap("Greys")


from src import utils

st.set_page_config(
    page_title="NAML - Music Genre Classifcation 🎼",
    page_icon="🎼"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

training_embeddings = st.cache_data(joblib.load)("models/GAP-data-aug-0_fold4_20230710_215510/training_embeddings.pkl")
pca = st.cache_data(joblib.load)("models/GAP-data-aug-0_fold4_20230710_215510/PCA.pkl")
scaler = st.cache_data(joblib.load)("models/GAP-data-aug-0_fold4_20230710_215510/StdScaler.pkl")
trues = st.cache_data(joblib.load)("models/GAP-data-aug-0_fold4_20230710_215510/training_labels.pkl")

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
genre_dict_colab = {
    0: 'rock',
    1: 'jazz',
    2: 'classical',
    3: 'pop',
    4: 'reggae',
    5: 'metal',
    6: 'disco',
    7: 'country',
    8: 'blues',
    9: 'hiphop'}

SAMPLE_RATE = 22_050
MODEL_DIR = "models/GAP-data-aug-0_fold4_20230710_215510"

st.image("images/logo_polimi.png")
st.divider()

st.subheader("NAML 2022: Practical project")
st.title("Automatic music genre classification with Deep Learning")


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
        wav, sr = load(audio_file)
        wav = Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
        print("wav_mean:", torch.mean(wav))
    elif url:
        try:
            wav, sr = load(utils.get_audio_stream_from_youtube(url))
        except RegexMatchError:
            st.error("URL was not resolved")

if thumbnail_url is not None:
    st.image(thumbnail_url, use_column_width=True)


if wav is not None:
    st.divider()
    st.markdown("")

    model, transform = utils.load_model_and_transform(MODEL_DIR, checkpoint="accuracy", model_type="CNN")
    model.eval()

    wav = wav[:1, :]

    st.audio(wav.numpy(), sample_rate=SAMPLE_RATE)

    spec = transform(wav).squeeze().numpy()
    offset = np.random.randint(spec.shape[-1] - 1200)
    spec = np.log(spec+1)[400::-1, offset:offset+1200]
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.axis(False)
    ax.imshow(spec)
    st.pyplot(fig)

    st.divider()
    left, right = st.columns(2)
    left.header("Inference")

    slices = utils.slice_audio(
        wav=wav,
        slice_duration=4.0,
        overlap=0.75,
        sample_rate=SAMPLE_RATE,
        max_duration=60.0
    )
    print("NUM SLICES:", len(slices))

    palplot = sns.palplot(sns.color_palette("Set3", n_colors=10))
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

    figure = st.empty()
    container = st.container()

    metrics_column, feature_column = container.columns((1, 4))
    feature_figure = feature_column.empty()
    metrics = [metrics_column.empty() for i in range(5)]

    proj = pca.transform(scaler.transform(training_embeddings))
    trues = [genre_dict[int(k)] for k in trues]
    training_feat_space = px.scatter_3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        color=trues,
        color_discrete_map={
            k: v for k, v in zip(genre_dict.values(), px.colors.qualitative.Set3)
        },
    )
    training_feat_space.update_traces(marker_size=5)
    feature_figure.plotly_chart(training_feat_space, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_prop_cycle('color', sns.color_palette('Set3', 10))
    fig.tight_layout()
    plt.axis(False)
    ax.set_xlim(0, len(slices)-1)

    activation = {}
    model.pool.register_forward_hook(utils.get_activation("embedding", activation))

    total_probas = torch.Tensor()
    total_feats = torch.Tensor()
    with torch.no_grad():
        for i, x in enumerate(slices):
            x = x.unsqueeze(0)
            x = transform(x)
            probas = model(x)

            probas -= torch.min(probas)
            probas **= 10
            probas /= torch.sum(probas)

            # probas = torch.nn.functional.softmax(probas, dim=1)
            feats = activation["embedding"]
            total_feats = torch.concatenate((total_feats, feats), dim=0)

            total_probas = torch.concatenate((total_probas, probas), dim=0)
            mean_probas = torch.mean(total_probas, dim=0).numpy() * 100

            top_classes = np.argsort(mean_probas)[::-1]

            ax.stackplot(torch.arange(i+1), *torch.vsplit(total_probas.T, 10))
            figure.pyplot(fig)

            for k, m in enumerate(metrics):
                m.metric(f"{genre_dict[top_classes[k]]}", f"{mean_probas[top_classes[k]]:5>.2f} %")

    ax.stackplot(torch.arange(i+1), *torch.vsplit(total_probas.T, 10))
    figure.pyplot(fig)

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
