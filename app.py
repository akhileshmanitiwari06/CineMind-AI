# app.py
# app.py (top) — graceful import handling for server (OpenCV import)
import streamlit as st
import tempfile, os

# Try to import core module; if cv2 (or any binary) fails to import on server,
# show a friendly message and the full traceback in an expander so we can debug.
try:
    from cinemind.core import analyze_video_cinematic, explanations_to_tts
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    st.set_page_config("CineMind AI", layout="wide")
    st.title("CineMind AI — Movie Scene Explainer (Starter)")
    st.error("The server failed to import a required binary (likely OpenCV). See full error below.")
    with st.expander("Full import error (copy this and paste to chat):"):
        st.code(tb)
    # Stop the app (prevents redacted crash)
    st.stop()

# If import succeeded, the rest of your app continues below...

st.title("CineMind AI — Movie Scene Explainer (Starter)")

uploaded = st.file_uploader("Upload movie clip (mp4, <=2min recommended)", type=["mp4","mov","avi"])
use_whisper = st.checkbox("Use Whisper for transcript (slow, optional)", value=False)
use_llm = st.checkbox("Use OpenAI to polish explanations (requires OPENAI_API_KEY)", value=False)
sample_fps = st.slider("Visual sampling fps (lower → faster)", min_value=0.5, max_value=2.0, value=1.0, step=0.5)
max_scenes = st.slider("Max scenes to analyze", min_value=1, max_value=20, value=6)

video_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush(); tmp.close()
    
    
    video_path = tmp.name
else:
    if st.button("Use example clip"):
        example = os.path.join("example_inputs","sample_clip.mp4")
        if os.path.exists(example):
            video_path = example
        else:
            st.warning("Add sample_clip.mp4 in example_inputs/ to use example.")

if video_path:
    with st.spinner("Analyzing clip (this may take a while depending on options)..."):
        results = analyze_video_cinematic(video_path, max_scenes=max_scenes, sample_fps=sample_fps, use_whisper=use_whisper, use_llm=use_llm)
    st.success("Analysis complete")
    for scene in results:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_no = int(scene['start'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if ret:
            try:
                display_frame = frame.copy()
                text = f"{scene['start']:.1f}s"
                org = (10, 30)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_frame, text, org, font, 0.9, (255,255,255), 2, cv2.LINE_AA)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            except Exception:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            st.image(Image.fromarray(frame_rgb),
                     caption=f"Scene {scene['scene_index']} thumbnail — {scene['start']:.1f}s to {scene['end']:.1f}s",
                     use_container_width=True)
        
                     
        
        st.markdown(f"### Scene {scene['scene_index']}  — {scene['start']:.1f}s to {scene['end']:.1f}s")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.write("**Visuals**")
            st.write(scene['visuals'])
        with col2:
            st.write("**Audio**")
            st.write(scene['audio'])
        with col3:
            st.write("**Emotions (agg)**")
            st.write(scene['emotions'].get('agg_emotion', {}))
        st.write("**Transcript (snippet)**")
        st.write(scene.get('transcript','(none)')[:500])
        st.write("**Cinematic explanation**")
        st.info(scene.get('explanation','(no explanation)'))
    # TTS combine
    explanations = [s.get('explanation','') for s in results]
    audio_bytes = explanations_to_tts(explanations, lang='en')
    if audio_bytes:
        st.audio(audio_bytes, format='audio/mp3')
# app.py
import streamlit as st
from cinemind.core import analyze_video_cinematic, explanations_to_tts
import tempfile, os
import cv2
from PIL import Image


st.set_page_config("CineMind AI", layout="wide")
st.title("CineMind AI — Movie Scene Explainer (Starter)")

uploaded = st.file_uploader("Upload movie clip (mp4, <=2min recommended)", type=["mp4","mov","avi"])
use_whisper = st.checkbox("Use Whisper for transcript (slow, optional)", value=False)
use_llm = st.checkbox("Use OpenAI to polish explanations (requires OPENAI_API_KEY)", value=False)
sample_fps = st.slider("Visual sampling fps (lower → faster)", min_value=0.5, max_value=2.0, value=1.0, step=0.5)
max_scenes = st.slider("Max scenes to analyze", min_value=1, max_value=20, value=6)

video_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush(); tmp.close()
    
    
    video_path = tmp.name
else:
    if st.button("Use example clip"):
        example = os.path.join("example_inputs","sample_clip.mp4")
        if os.path.exists(example):
            video_path = example
        else:
            st.warning("Add sample_clip.mp4 in example_inputs/ to use example.")

if video_path:
    with st.spinner("Analyzing clip (this may take a while depending on options)..."):
        results = analyze_video_cinematic(video_path, max_scenes=max_scenes, sample_fps=sample_fps, use_whisper=use_whisper, use_llm=use_llm)
    st.success("Analysis complete")
    for scene in results:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_no = int(scene['start'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if ret:
            try:
                display_frame = frame.copy()
                text = f"{scene['start']:.1f}s"
                org = (10, 30)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_frame, text, org, font, 0.9, (255,255,255), 2, cv2.LINE_AA)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            except Exception:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            st.image(Image.fromarray(frame_rgb),
                     caption=f"Scene {scene['scene_index']} thumbnail — {scene['start']:.1f}s to {scene['end']:.1f}s",
                     use_container_width=True)
        
                     
        
        st.markdown(f"### Scene {scene['scene_index']}  — {scene['start']:.1f}s to {scene['end']:.1f}s")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.write("**Visuals**")
            st.write(scene['visuals'])
        with col2:
            st.write("**Audio**")
            st.write(scene['audio'])
        with col3:
            st.write("**Emotions (agg)**")
            st.write(scene['emotions'].get('agg_emotion', {}))
        st.write("**Transcript (snippet)**")
        st.write(scene.get('transcript','(none)')[:500])
        st.write("**Cinematic explanation**")
        st.info(scene.get('explanation','(no explanation)'))
    # TTS combine
    explanations = [s.get('explanation','') for s in results]
    audio_bytes = explanations_to_tts(explanations, lang='en')
    if audio_bytes:
        st.audio(audio_bytes, format='audio/mp3')
