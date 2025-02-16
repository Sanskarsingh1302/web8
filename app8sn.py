import cv2
import streamlit as st
from ultralytics import YOLO
import torch
import yagmail
from collections import Counter
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load YOLOv8n model

# Email configuration
sender_email = "sanskarsingh725@gmail.com"
receiver_emails = st.text_area("Enter recipient email addresses (comma-separated)").split(',')
yag = yagmail.SMTP(user=sender_email, password='nxwf lhvx otdp jqsg')

# Function to send email notification
def send_email_notification(detected_object, subject="Object Detection Alert"):
    content = f"A {detected_object} was detected."
    yag.send(to=receiver_emails, subject=subject, contents=content)
    st.success(f"Email notification sent for detected object: {detected_object}")

# Function to detect available cameras
def available_cameras():
    available_cam_indexes = []
    for i in range(10):  # Test for the first 10 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cam_indexes.append(i)
            cap.release()
    return available_cam_indexes

# Function to detect objects in a video
def detect_objects_in_video(video_path, confidence_threshold):
    detected_objects = Counter()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_frames_dir = "detected_frames"
    os.makedirs(detected_frames_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame to speed up detection
            results = model(frame)  # Run YOLOv8 inference

            for result in results:  # Loop through results
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    if confidence < confidence_threshold:
                        continue  # Skip low-confidence detections

                    label = model.names[class_id]  # Class label
                    detected_objects[label] += 1

                    # Save frame with detection
                    detected_frame_path = os.path.join(detected_frames_dir, f"frame_{frame_count}_{label}.jpg")
                    cv2.imwrite(detected_frame_path, frame)

    cap.release()
    return detected_objects

# Function to compare objects from two videos
def compare_objects(before_objects, after_objects):
    new_objects = {}
    for obj, count in after_objects.items():
        if obj not in before_objects or after_objects[obj] > before_objects[obj]:
            new_objects[obj] = after_objects[obj] - before_objects.get(obj, 0)
    return new_objects

# Function to generate and download CSV report
def export_as_csv(detected_objects):
    csv_path = "/mnt/data/detected_objects.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Object', 'Count'])
        for obj, count in detected_objects.items():
            writer.writerow([obj, count])
    return csv_path

# Streamlit app
st.title("SCHOOL OF AERONAUTICAL ENGINEERING PRESENTS Live video based Object Detection")

# Get available cameras
available_cams = available_cameras()
if available_cams:
    camera_index = st.selectbox("Select a Camera", available_cams)
else:
    st.warning("No camera found!")
    st.stop()

# Option for Live Detection or Compare Two Videos
detection_option = st.radio("Select Mode", ("Live Detection", "Compare Two Videos"))

# Object selection for alerts
object_to_alert = st.multiselect("Select objects to receive alerts for", 
                                 ['person', 'car', 'bus', 'bicycle', 'truck', 'cat', 'dog'])

# Variable to store detected objects
detected_objects = Counter()

# Confidence threshold slider
confidence_threshold = st.slider("Set Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)

if detection_option == "Live Detection":
    # Start/Stop button for Live Detection
    start_detection = st.button("Start Live Detection")
    stop_detection = st.button("Stop Live Detection", disabled=not start_detection)

    if start_detection and not stop_detection:
        # Video capture with the selected camera
        cap = cv2.VideoCapture(camera_index)

        # Streamlit video stream display
        stframe = st.empty()

        # Detection loop
        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture video.")
                break

            # Run YOLOv8 inference on the frame
            results = model(frame)  # Perform inference

            # Extract predictions
            for result in results:  # Loop through each result in the batch (only 1 image in this case)
                boxes = result.boxes  # Extract boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
                    confidence = box.conf[0]  # Confidence score
                    class_id = int(box.cls[0])  # Class ID of detected object
                    label = model.names[class_id]  # Class label

                    if confidence >= confidence_threshold:
                        # Draw bounding boxes and labels on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Count detected objects
                        detected_objects[label] += 1

                        # Send email if the detected object is in the selected list
                        if label in object_to_alert:
                            send_email_notification(label)

            # Show the frame in Streamlit
            stframe.image(frame, channels="BGR")

        cap.release()

    # Report generation
    if st.button("Generate Report") and detected_objects:
        report = "Object Detection Report:\n"
        for obj, count in detected_objects.items():
            report += f"{obj}: {count}\n"

        # Save the report to a file
        report_file_path = "/mnt/data/detection_report.txt"
        with open(report_file_path, "w") as f:
            f.write(report)

        st.success("Report generated and saved as detection_report.txt.")

        # Download the report
        with open(report_file_path, "rb") as report_file:
            st.download_button("Download Report", report_file, file_name="detection_report.txt")

        # Export as CSV
        csv_path = export_as_csv(detected_objects)
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", f, file_name="detected_objects.csv")

        # Plot detected objects count
        st.subheader("Detected Objects Count")
        fig, ax = plt.subplots()
        ax.bar(detected_objects.keys(), detected_objects.values())
        st.pyplot(fig)

elif detection_option == "Compare Two Videos":
    before_video_path = st.file_uploader("Upload Before Video", type=["mp4", "avi", "mov"])
    after_video_path = st.file_uploader("Upload After Video", type=["mp4", "avi", "mov"])

    if before_video_path and after_video_path:
        # Temporary files for uploaded videos
        before_video_temp = tempfile.NamedTemporaryFile(delete=False)
        before_video_temp.write(before_video_path.read())
        after_video_temp = tempfile.NamedTemporaryFile(delete=False)
        after_video_temp.write(after_video_path.read())

        # Detect objects in the before and after videos
        st.write("Detecting objects in 'Before' video...")
        before_objects = detect_objects_in_video(before_video_temp.name, confidence_threshold)
        st.write("Detecting objects in 'After' video...")
        after_objects = detect_objects_in_video(after_video_temp.name, confidence_threshold)

        # Compare the two sets of detected objects
        st.write("Comparing objects between 'Before' and 'After' videos...")
        new_objects = compare_objects(before_objects, after_objects)

        # Display the comparison result
        st.write("New objects detected or increased count in 'After' video:")
        for obj, count in new_objects.items():
            st.write(f"{obj}: {count}")

        # Plot new objects count
        if new_objects:
            fig, ax = plt.subplots()
            ax.bar(new_objects.keys(), new_objects.values())
            ax.set_title("New Objects Detected or Increased Count in 'After' Video")
            ax.set_xlabel("Objects")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Export the result as CSV
        csv_path = export_as_csv(new_objects)
        with open(csv_path, "rb") as f:
            st.download_button("Download Comparison CSV", f, file_name="new_objects_comparison.csv")

