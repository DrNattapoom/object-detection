import streamlit as st
import streamlit_webrtc as stweb
import tensorflow as tf
import utils as ut

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def main() -> None:
    """
    Object detection using deep learning with convolutional neural network model.
    """
    st.title("Object Detection")
    st.caption(
        "Object Detection using Deep Learning with Convolutional Neural Network Model"
    )
    # load the trained model
    model = tf.keras.models.load_model(
        "model/object_detector.h5",
        compile=False
    )
    with st.sidebar:
        options = {
            0: "Webcam",
            1: "Video File",
        }
        selected = st.selectbox(
            "Select the input source",
            options.keys(),
            index=1,
            format_func=lambda key: options[key]
        )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    if selected == 0:
        class VideoProcessor:
            def recv(self, frame):
                return ut.frame_callback(frame, model, 256, confidence_threshold)
        stweb.webrtc_streamer(
            key="key",
            video_processor_factory=VideoProcessor,
            rtc_configuration=stweb.RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

    elif selected == 1:
        uploaded_video = st.file_uploader(
            "Upload Video",
            type=['mp4', 'mpeg', 'mov']
        )

        if uploaded_video != None:
            vid = uploaded_video.name
            with open(vid, mode='wb') as f:
                # save video to disk
                f.write(uploaded_video.read())
            st_video = open(vid, 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            ut.detect(vid, model, threshold=confidence_threshold,
                      with_output=True)
            st_video = open('detected_video.mp4', 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video")


if __name__ == "__main__":
    main()
