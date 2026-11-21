
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd
import plotly.express as px
import cv2
import tempfile
import os
from collections import Counter

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WarehouseVision AI",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "warehouse_yolo11_final.pt"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the warehouse model: {error}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📦 WarehouseVision AI")
st.subheader(
    "Autonomous Warehouse Perception & Logistics Optimization"
)
st.caption(
    "Computer Vision • YOLO11 • ByteTrack • Warehouse Analytics"
)

st.divider()

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU / NMS Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

st.sidebar.caption(
    "Higher confidence filters weaker predictions. "
    "Lower IoU generally suppresses more overlapping boxes."
)

st.sidebar.divider()
st.sidebar.success("Warehouse YOLO11 model loaded")

# --------------------------------------------------
# ANALYTICS DISPLAY
# --------------------------------------------------

def show_analytics(counts, title):

    st.header(title)

    total = sum(counts.values())
    categories = len(counts)

    most_detected = (
        max(counts, key=counts.get).title()
        if counts else "None"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Objects", total)
    col2.metric("Object Categories", categories)
    col3.metric("Most Detected Class", most_detected)

    if not counts:
        st.warning("No objects detected.")
        return

    df = pd.DataFrame(
        list(counts.items()),
        columns=["Object", "Count"]
    ).sort_values("Count", ascending=False)

    left, right = st.columns(2)

    with left:
        st.subheader("📋 Object Counts")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.subheader("📊 Object Distribution")

        fig = px.bar(
            df,
            x="Object",
            y="Count",
            color="Object",
            text="Count",
            title="AI-Detected Objects by Class"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.download_button(
        "⬇️ Download Analytics CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="warehouse_analytics.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

image_tab, video_tab, about_tab = st.tabs(
    ["📷 Image Analysis", "🎥 Video Tracking", "ℹ️ About"]
)

with image_tab:

    st.header("📷 Warehouse Image Analysis")

    uploaded_image = st.file_uploader(
        "Upload a warehouse image",
        type=["jpg", "jpeg", "png"],
        key="warehouse_image"
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded Warehouse Image",
            use_container_width=True
        )

        if st.button("🔍 Analyze Image", type="primary"):

            with st.spinner("Detecting warehouse objects..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    iou=iou_threshold,
                    max_det=100,
                    verbose=False
                )

            result = results[0]
            counts = Counter()
            detection_records = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                score = float(box.conf[0])

                counts[class_name] += 1

                detection_records.append({
                    "Object": class_name,
                    "Confidence": round(score, 4)
                })

            annotated_image = result.plot()
            annotated_rgb = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            st.success("Image analysis completed!")

            st.subheader("🎯 Detection Results")

            st.image(
                annotated_rgb,
                caption="YOLO11 Annotated Image",
                use_container_width=True
            )

            image_buffer = tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
            )

            try:
                Image.fromarray(annotated_rgb).save(
                    image_buffer.name
                )

                with open(image_buffer.name, "rb") as file:
                    st.download_button(
                        "⬇️ Download Annotated Image",
                        data=file.read(),
                        file_name="warehouse_detection.png",
                        mime="image/png"
                    )
            finally:
                image_buffer.close()
                os.unlink(image_buffer.name)

            show_analytics(
                counts,
                "📊 Warehouse Image Analytics"
            )

            if detection_records:
                st.subheader("🔎 Individual Detection Details")

                st.dataframe(
                    pd.DataFrame(detection_records),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Counts represent AI predictions, not verified "
                "physical inventory. Occlusion and overlapping "
                "objects may cause missed or duplicate detections."
            )

# --------------------------------------------------
# VIDEO TRACKING
# --------------------------------------------------

with video_tab:

    st.header("🎥 Warehouse Video Tracking")

    uploaded_video = st.file_uploader(
        "Upload a warehouse video",
        type=["mp4", "avi", "mov"],
        key="warehouse_video"
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button("🚀 Start Video Tracking", type="primary"):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as input_file:

                    input_file.write(uploaded_video.getbuffer())
                    input_path = input_file.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Could not open the uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_file = tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False
                )

                output_path = output_file.name
                output_file.close()

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    st.error("Could not create the output video.")
                    st.stop()

                unique_tracks = set()
                untracked_counts = Counter()
                frame_counts = Counter()
                processed_frames = 0

                progress = st.progress(0)
                status = st.empty()

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    result = model.track(
                        source=frame,
                        persist=True,
                        conf=confidence,
                        iou=iou_threshold,
                        tracker="bytetrack.yaml",
                        max_det=100,
                        verbose=False
                    )[0]

                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]

                            frame_counts[class_name] += 1

                            if box.id is not None:
                                track_id = int(box.id[0])
                                unique_tracks.add(
                                    (class_name, track_id)
                                )
                            else:
                                untracked_counts[class_name] += 1

                    annotated_frame = result.plot()
                    writer.write(annotated_frame)

                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        if total_frames > 0:
                            progress.progress(
                                min(
                                    processed_frames / total_frames,
                                    1.0
                                )
                            )

                        status.write(
                            f"Processed frames: {processed_frames}"
                        )

                cap.release()
                cap = None

                writer.release()
                writer = None

                progress.progress(1.0)
                status.success(
                    f"Tracking completed! Frames processed: "
                    f"{processed_frames}"
                )

                track_counts = Counter(
                    class_name
                    for class_name, track_id in unique_tracks
                )

                for name, count in untracked_counts.items():
                    track_counts[name] += count

                st.subheader("🎬 Tracked Video")

                st.video(output_path)

                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "⬇️ Download Tracked Video",
                        data=video_file.read(),
                        file_name="warehouse_tracked.mp4",
                        mime="video/mp4"
                    )

                show_analytics(
                    track_counts,
                    "📊 Video Tracking Analytics"
                )

                st.caption(
                    "Tracked counts are unique track IDs where "
                    "available. Objects without a tracking ID are "
                    "counted as detection observations. Track IDs "
                    "can change when objects disappear or are occluded."
                )

            except Exception as error:
                st.error(f"Video processing failed: {error}")

            finally:

                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.unlink(input_path)

                if output_path and os.path.exists(output_path):
                    # Keep output available during this Streamlit run.
                    pass

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

with about_tab:

    st.header("ℹ️ About WarehouseVision AI")

    st.markdown("""
    **WarehouseVision AI** is an intelligent warehouse monitoring
    application powered by a custom-trained YOLO11 model.

    ### Core capabilities

    - Warehouse object detection
    - Person, forklift, box and pallet recognition
    - Real-time object tracking in uploaded videos
    - Object counting and distribution analytics
    - Annotated image and video downloads
    - CSV analytics export

    ### Technology stack

    - Python
    - YOLO11 and Ultralytics
    - PyTorch
    - OpenCV
    - ByteTrack
    - Streamlit
    - Pandas
    - Plotly

    ### Model information

    - Architecture: YOLO11n
    - Dataset: Warehouse Objects
    - Classes: 9
    - Test mAP@50: 63.92%
    - Test mAP@50-95: 36.42%

    Model performance depends on image quality, object visibility,
    lighting, camera angle and confidence settings.
    """)

st.divider()

st.caption(
    "WarehouseVision AI | Computer Vision & Intelligent Logistics"
)
