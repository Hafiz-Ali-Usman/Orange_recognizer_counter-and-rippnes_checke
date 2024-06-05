import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
from io import BytesIO

# Streamlit App Title
st.title("Orange Detection and Ripeness Analysis")

# Cached function to load the model and processor
@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return processor, model

processor, model = load_model()

# Function to check color for ripeness analysis
def is_ripe(color):
    avg_red = color[0] / 255.0
    return avg_red > 0.5

# Function to visualize detection results
def visualize_results(image, results, class_labels):
    fig, ax = plt.subplots()
    ax.imshow(image)

    total_objects = len(results["scores"])
    ripe_count = 0
    unripe_count = 0

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{class_labels[label.item()]}: {round(score.item(), 3)}"

        # Draw bounding box
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Display label text
        plt.text(box[0], box[1], label_text, color='b', verticalalignment='bottom')

        # Analyze ripeness based on average color
        region = image.crop((box[0], box[1], box[2], box[3]))
        region_array = np.array(region)
        avg_color = np.mean(region_array, axis=(0, 1))

        # Determine ripeness and add annotation
        if is_ripe(avg_color):
            ripe_count += 1
            plt.text(box[0], box[3] + 10, "Ripe", color='g', fontsize=8, verticalalignment='bottom')
        else:
            unripe_count += 1
            plt.text(box[0], box[3] + 10, "Not Ripe", color='r', fontsize=8, verticalalignment='bottom')

    plt.axis('off')
    plt.tight_layout()
    
    return fig, total_objects, ripe_count, unripe_count

# File uploader to load images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Ensure an image is uploaded
if uploaded_file:
    try:
        # Load the image
        image = Image.open(uploaded_file)

        # Process the image and run object detection
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process detection results
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Class labels for visualization
        class_labels = {i: model.config.id2label[i] for i in model.config.id2label.keys()}

        # Visualize results and perform ripeness analysis
        fig, total_objects, ripe_count, unripe_count = visualize_results(image, results, class_labels)

        # Display visualization
        st.pyplot(fig)

        # Display summary information
        st.write(f"Total number of objects detected: {total_objects}")
        st.write(f"Number of ripe objects: {ripe_count}")
        st.write(f"Number of unripe objects: {unripe_count}")
    
    except Exception as e:
        # Error handling
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload an image to start analysis.")
