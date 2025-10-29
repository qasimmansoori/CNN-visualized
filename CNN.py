import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
class_names = [str(i) for i in range(10)]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)
    def forward_features(self, x):
        f1 = self.block1(x); f2 = self.block2(f1); f3 = self.block3(f2); f4 = self.block4(f3)
        return [f1, f2, f3, f4]
    def forward(self, x):
        feats = self.forward_features(x)
        out = self.fc(self.flatten(self.gap(feats[-1])))
        return feats, out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
ckpt = torch.load("best_model.pt", map_location=device)
if any(k.split(".")[0].isdigit() for k in ckpt.keys()):
    mapping = {
        "0.": "block1.0.", "1.": "block1.1.",
        "3.": "block2.0.", "4.": "block2.1.",
        "6.": "block3.0.", "7.": "block3.1.",
        "9.": "block4.0.", "10.": "block4.1.",
        "15.": "fc."
    }
    new_state = {}
    for k, v in ckpt.items():
        mapped = False
        for src, dst in mapping.items():
            if k.startswith(src):
                new_state[k.replace(src, dst, 1)] = v
                mapped = True
                break
        if not mapped: new_state[k] = v
    model.load_state_dict(new_state, strict=False)
else:
    model.load_state_dict(ckpt)
model.to(device).eval()

st.markdown("<div style='text-align:left; margin:6px 0 6px 20px;'>\
<span style=\"font-size:28px; font-weight:700;\">🖌️ Handwritten Digit Recognition + CNN Visualization</span>\
<div style=\"font-size:12px; margin-top:4px; color:gray;\">Draw a digit and see CNN feature extraction live.</div>\
</div>", unsafe_allow_html=True)

if "last_hash" not in st.session_state: st.session_state["last_hash"] = None
left_col, mid_col, right_col = st.columns([1, 1, 1])

def visualize_blocks_grid(features, block_names, num_features=4):
    rows, cols = num_features, len(features)
    fig = make_subplots(rows=rows, cols=cols)
    for c, feat in enumerate(features):
        t = feat.squeeze(0).detach().cpu()
        for r in range(rows):
            img = np.flipud(t[r].numpy()) if r < t.shape[0] else np.zeros((t.shape[1], t.shape[2]))
            fig.add_trace(go.Heatmap(z=img, colorscale='Viridis', showscale=False), row=r+1, col=c+1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=110*rows, margin=dict(t=5,b=5,l=5,r=5))
    st.plotly_chart(fig, use_container_width=True)

features = probs = pred_class = None
with left_col:
    st.markdown("<h3 style='margin-top:-6px;'>Draw here</h3>", unsafe_allow_html=True)
    canvas = st_canvas(fill_color="white", stroke_width=10, stroke_color="black",
                       background_color="white", width=280, height=280, drawing_mode="freedraw", key="canvas")
    if canvas and canvas.image_data is not None:
        img_arr = (255 - canvas.image_data[:, :, 0]).astype(np.uint8)
        img_hash = hashlib.md5(img_arr.tobytes()).hexdigest()
        if img_hash != st.session_state["last_hash"]:
            st.session_state["last_hash"] = img_hash
            img = Image.fromarray(img_arr).resize((28, 28)).convert("L")
            input_tensor = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
            ])(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features, output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                pred_class = int(np.argmax(probs))
        with mid_col:
            st.markdown("<h3 style='text-align:center;margin-top:-6px;margin-bottom:6px;'>🔍 CNN Feature Extraction</h3>", unsafe_allow_html=True)
            visualize_blocks_grid(features, ["Block 1", "Block 2", "Block 3", "Block 4"])

with right_col:
    if probs is not None:
        st.subheader(f"🧠 Predicted Digit: **{class_names[pred_class]}**")
        df = pd.DataFrame({"Digit": class_names, "Probability (%)": probs*100}).sort_values("Probability (%)", ascending=False)
        fig = px.bar(df, x="Digit", y="Probability (%)", color="Digit",
                     color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_layout(showlegend=False, height=260, margin=dict(t=10,b=10,l=10,r=10))
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start drawing to visualize CNN features and predictions.")
