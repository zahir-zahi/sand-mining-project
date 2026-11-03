import numpy as np
import cv2
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Illegal Sand Mining Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("Illegal Sand Mining Detection")
st.subheader("Using Satellite Images")

st.write(
    "Our project uses satellite images from different "
    "dates to identify possible sand mining in river areas."
)

st.info("Satellite data source: Sentinel-2")

st.header("Upload Satellite Images")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Older Image")
    old_image = st.file_uploader(
        "Upload the older image",
        type=["png", "jpg", "jpeg"],
        key="old"
    )

with col2:
    st.subheader("Newer Image")
    new_image = st.file_uploader(
        "Upload the newer image",
        type=["png", "jpg", "jpeg"],
        key="new"
    )

if old_image:
    st.image(old_image, caption="Older Image")

if new_image:
    st.image(new_image, caption="Newer Image")

st.header("Detection Results")

if old_image and new_image:
    st.success("Both images uploaded successfully!")

    img1 = Image.open(old_image).convert("RGB")
    img2 = Image.open(new_image).convert("RGB")
    img2 = img2.resize(img1.size)

    a = np.array(img1)
    b = np.array(img2)

    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    changed = np.count_nonzero(mask) / mask.size * 100

    st.metric("Changed area", f"{changed:.2f}%")
    st.image(mask, caption="Difference map (white = changed area)")
else:
    st.write("Upload both images to get started.")

st.caption("Student Project | Zahir and Team")