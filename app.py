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
    col1.image(old_image, caption="Older Image", use_container_width=True)

if new_image:
    col2.image(new_image, caption="Newer Image", use_container_width=True)

# Settings the user can change
st.sidebar.header("Detection Settings")
sensitivity = st.sidebar.slider(
    "Sensitivity (lower = finds more changes)", 10, 100, 24
)
min_area = st.sidebar.slider(
    "Smallest patch to show (bigger = fewer small dots)", 10, 2000, 661
)

st.header("Detection Results")

if old_image and new_image:
    st.success("Both images uploaded successfully!")

    img1 = Image.open(old_image).convert("RGB")
    img2 = Image.open(new_image).convert("RGB")
    img2 = img2.resize(img1.size)

    a = np.array(img1)
    b = np.array(img2)

    # 1. Blur both images to remove tiny noise
    a_blur = cv2.GaussianBlur(a, (7, 7), 0)
    b_blur = cv2.GaussianBlur(b, (7, 7), 0)

    # 2. Find the difference
    diff = cv2.absdiff(a_blur, b_blur)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)

    # 3. Clean the mask: remove small dots, join nearby pieces
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_big = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

    # 4. Keep only big patches
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, big_contours, -1, 255, thickness=-1)

    # 5. Paint changed areas red on the new image
    result = b.copy()
    red = np.zeros_like(b)
    red[:, :] = (255, 0, 0)
    blended = cv2.addWeighted(b, 0.4, red, 0.6, 0)
    result[clean_mask > 0] = blended[clean_mask > 0]
    cv2.drawContours(result, big_contours, -1, (255, 0, 0), 2)

    changed = np.count_nonzero(clean_mask) / clean_mask.size * 100

    m1, m2 = st.columns(2)
    m1.metric("Changed area", f"{changed:.2f}%")
    m2.metric("Suspicious patches found", len(big_contours))

    r1, r2 = st.columns(2)
    r1.image(
        result,
        caption="Suspicious areas in red (on the newer image)",
        use_container_width=True
    )
    r2.image(
        clean_mask,
        caption="Cleaned difference map (white = changed area)",
        use_container_width=True
    )

    st.caption(
        "Note: red areas are only possible sand mining. "
        "Please check them with a field visit or a clearer image."
    )
else:
    st.write("Upload both images to get started.")

st.caption("Student Project | Zahir and Team")
