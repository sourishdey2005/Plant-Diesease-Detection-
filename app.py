from models import resnext50_32x4d
from dataset import CassavaDataset, get_transforms, classes
from inference import load_state, inference
from utils import CFG
from grad_cam import SaveFeatures, getCAM, plotGradCAM
import os
import gc
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from html_mardown import (
    app_off,
    app_off2,
    model_predicting,
    loading_bar,
    result_pred,
    image_uploaded_success,
    more_options,
    class0,
    class1,
    class2,
    class3,
    class4,
    s_load_bar,
    class0_side,
    class1_side,
    class2_side,
    class3_side,
    class4_side,
    unknown,
    unknown_side,
    unknown_w,
    unknown_msg,
)

# Enable garbage collection
gc.enable()

# Hide warnings
# (Removed deprecated Streamlit option `deprecation.showfileUploaderEncoding`)
# st.set_option("deprecation.showfileUploaderEncoding", False)

import warnings
warnings.filterwarnings("ignore")

# Configure page layout
st.set_page_config(
    page_title="Cassava Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .disease-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .healthy-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .info-box {
        background-color: #f0f4ff;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    .header-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subheader-text {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set App title with custom styling
st.markdown('<div class="header-title">🌿 Cassava Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">AI-Powered Plant Health Analysis</div>', unsafe_allow_html=True)


# Set the directory path
my_path = "."

# Load sample data with error handling
try:
    test = pd.read_csv(my_path + "/data/sample.csv")
except FileNotFoundError:
    st.error("❌ Error: `data/sample.csv` file not found!")
    st.info("Please ensure the following files exist in your project:")
    st.write("- `data/sample.csv`")
    st.write("- `images/img_1.jpg`, `images/img_2.jpg`, `images/img_3.jpg`")
    st.write("- `images/banner.png`")
    st.stop()

img_1_path = my_path + "/images/img_1.jpg"
img_2_path = my_path + "/images/img_2.jpg"
img_3_path = my_path + "/images/img_3.jpg"
banner_path = my_path + "/images/banner.png"
output_image = my_path + "/images/gradcam2.png"

# Verify image files exist
for img_path in [img_1_path, img_2_path, img_3_path, banner_path]:
    if not os.path.exists(img_path):
        st.warning(f"⚠️ Warning: {img_path} not found")


# Create sidebar with banner
with st.sidebar:
    st.image(banner_path, use_column_width=True)
    st.markdown("---")
    st.markdown("### 📊 Quick Guide")
    st.info("1. Choose a demo image or upload your own\n2. Wait for analysis\n3. Review results and Grad-CAM visualization")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📚 About", "ℹ️ How It Works"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🌾 Demo Images")
        st.markdown("*Select a sample cassava leaf image*")
        menu = ["Select an Image", "Image 1", "Image 2", "Image 3"]
        choice = st.selectbox("Choose demo image:", menu, label_visibility="collapsed")
    
    with col2:
        st.markdown("### 📸 Upload Your Image")
        st.markdown("*or upload your own cassava leaf photo*")
        uploaded_image = st.file_uploader(
            "Choose an image (JPG or PNG)", type=["jpg", "png"], label_visibility="collapsed"
        )

with tab2:
    st.markdown("""
    ### About This Application
    
    This Cassava Disease Detection AI leverages deep learning to identify diseases in cassava plants.
    
    **Dataset:** [Kaggle - Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification/data)
    
    **Model Architecture:** ResNeXt50 (32x4d) with Grad-CAM visualization
    
    **Diseases Detected:**
    - 🦠 **CBB** - Cassava Bacterial Blight
    - 🌀 **CBSD** - Cassava Brown Streak Disease
    - 🟢 **CGM** - Cassava Green Mottle
    - 🔄 **CMD** - Cassava Mosaic Disease
    - ✅ **Healthy** - No disease detected
    """)

with tab3:
    st.markdown("""
    ### How Grad-CAM Works
    
    **Grad-CAM (Gradient-weighted Class Activation Map)** is a technique that highlights the regions 
    in an image that influenced the model's prediction.
    
    - 🔴 **Red areas** indicate regions the model focused on
    - 🟢 **Green areas** indicate less relevant regions
    - The visualization helps verify that the model is making decisions based on the correct plant features
    
    **Confidence Threshold:** Predictions below 57% confidence are marked as "Unknown"
    """)

st.markdown("---")


# DataLoader for pytorch dataset
def Loader(img_path=None, uploaded_image=None, upload_state=False, demo_state=True):
    test_dataset = CassavaDataset(
        test,
        img_path,
        uploaded_image=uploaded_image,
        transform=get_transforms(data="valid"),
        uploaded_state=upload_state,
        demo_state=demo_state,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    return test_loader


# Function to deploy the model and print the report
def deploy(file_path=None, uploaded_image=uploaded_image, uploaded=False, demo=True):
    # Load the model and the weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(CFG.model_name, pretrained=False)
    states = [load_state(my_path + "/weights/resnext50_32x4d_fold0_best.pth")]

    # For Grad-cam features
    final_conv = model.model.layer4[2]._modules.get("conv3")
    fc_params = list(model.model._modules.get("fc").parameters())

    # Display the uploaded/selected image
    st.markdown("---")
    
    with st.spinner("🔄 Analyzing image..."):
        if demo:
            test_loader = Loader(img_path=file_path)
            image_1 = cv2.imread(file_path)
        if uploaded:
            test_loader = Loader(
                uploaded_image=uploaded_image, upload_state=True, demo_state=False
            )
            image_1 = file_path
        st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
        st.sidebar.image(image_1, width=301, channels="BGR")

        for img in test_loader:
            activated_features = SaveFeatures(final_conv)
            # Save weight from fc
            weight = np.squeeze(fc_params[0].cpu().data.numpy())

            # Inference
            logits, output = inference(model, states, img, device)
            pred_idx = output.to("cpu").numpy().argmax(1)

            # Grad-cam heatmap display
            heatmap = getCAM(activated_features.features, weight, pred_idx)

            ##Reverse the pytorch normalization
            MEAN = torch.tensor([0.485, 0.456, 0.406])
            STD = torch.tensor([0.229, 0.224, 0.225])
            image = img[0] * STD[:, None, None] + MEAN[:, None, None]

            # Display image + heatmap
            plt.imshow(image.permute(1, 2, 0))
            plt.imshow(
                cv2.resize(
                    (heatmap * 255).astype("uint8"),
                    (328, 328),
                    interpolation=cv2.INTER_LINEAR,
                ),
                alpha=0.4,
                cmap="jet",
            )
            plt.savefig(output_image)
            plt.close()

            # Create results layout
            results_col1, results_col2 = st.columns([1, 1], gap="large")
            
            with results_col1:
                st.markdown("### 🔍 Prediction Result")
                
                # Display Unknown class if the highest probability is lower than 0.57
                if np.amax(logits) < 0.57:
                    st.warning("⚠️ **UNKNOWN CLASS** - All classes have low confidence", icon="⚠️")
                    st.info("The model is uncertain about this image. Please provide a clearer image of a cassava leaf.")
                # Display the class predicted if the highest probability is higher than 0.57
                else:
                    confidence = np.amax(logits) * 100
                    
                    if pred_idx[0] == 0:
                        st.error("🦠 **Cassava Bacterial Blight (CBB)**", icon="🦠")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    elif pred_idx[0] == 1:
                        st.error("🌀 **Cassava Brown Streak Disease (CBSD)**", icon="🌀")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    elif pred_idx[0] == 2:
                        st.warning("🟢 **Cassava Green Mottle (CGM)**", icon="🟢")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    elif pred_idx[0] == 3:
                        st.warning("🔄 **Cassava Mosaic Disease (CMD)**", icon="🔄")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    elif pred_idx[0] == 4:
                        st.success("✅ **Healthy - No Disease Detected**", icon="✅")
                        st.write(f"**Confidence:** {confidence:.2f}%")
            
            with results_col2:
                st.markdown("### 📊 Confidence Scores")
                classes_copy = classes.copy()
                classes_copy["Confidence %"] = logits.reshape(-1).tolist()
                classes_copy["Confidence %"] = classes_copy["Confidence %"] * 100
                
                # Create a bar chart for better visualization
                chart_data = classes_copy.set_index("class name")["Confidence %"]
                st.bar_chart(chart_data)

            # Display the Grad-Cam image
            st.markdown("### 🔥 Grad-CAM Visualization")
            st.write(
                "**Grad-CAM (Class Activation Map)** highlights the regions that influenced the model's decision. "
                "This helps verify that the model is focusing on relevant plant features."
            )
            
            grad_cam_col1, grad_cam_col2 = st.columns(2)
            with grad_cam_col1:
                st.write("**Original Image with Heatmap**")
                gram_im = cv2.imread(output_image)
                st.image(gram_im, use_column_width=True, channels="RGB")
            
            with grad_cam_col2:
                st.write("**Confidence Breakdown**")
                if np.amax(logits) < 0.57:
                    st.info("🔍 Model confidence is below threshold. Consider this prediction as uncertain.")
                classes_copy["class name"] = classes_copy["class name"].str.upper()
                classes_proba = classes_copy.style.background_gradient(cmap="RdYlGn")
                st.dataframe(classes_proba, use_container_width=True)
            
            del (
                model,
                states,
                fc_params,
                final_conv,
                test_loader,
                image_1,
                activated_features,
                weight,
                heatmap,
                gram_im,
                logits,
                output,
                pred_idx,
            )
            gc.collect()


# Set red flag if no image is selected/uploaded
if uploaded_image is None and choice == "Select an Image":
    st.sidebar.markdown(app_off, unsafe_allow_html=True)
    st.sidebar.markdown(app_off2, unsafe_allow_html=True)


# Deploy the model if the user uploads an image
if uploaded_image is not None:
    # Close the demo
    choice = "Select an Image"
    # Deploy the model with the uploaded image
    deploy(uploaded_image, uploaded=True, demo=False)
    del uploaded_image


# Deploy the model if the user selects Image 1
if choice == "Image 1":
    deploy(img_1_path)


# Deploy the model if the user selects Image 2
if choice == "Image 2":
    deploy(img_2_path)


# Deploy the model if the user selects Image 3
if choice == "Image 3":
    deploy(img_3_path)
