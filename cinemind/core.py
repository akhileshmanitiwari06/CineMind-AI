# cinemind/core.py
import os, io, math, tempfile
from typing import List, Dict
import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import librosa
import soundfile as sf
from gtts import gTTS
from dotenv import load_dotenv
load_dotenv()
openai = None  
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_KEY:
        import openai
        openai.api_key = OPENAI_KEY
except Exception:
    openai = None

# Optional face emotion library
try:
    from deepface import DeepFace
    have_deepface = True
except Exception:
    have_deepface = False

# -------------------------
# Utilities
# -------------------------
def ensure_ffmpeg():
    # If ffmpeg not on PATH, user should install it. We assume it's present.
    return True

# -------------------------
# Scene detection (PySceneDetect)
# -------------------------
def detect_scenes(video_path: str, threshold=30.0):
    """
    Returns list of scenes as (start_time_s, end_time_s)
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=15.0))
    base_timecode = video_manager.get_base_timecode()

    try:
        video_manager.set_downscale_factor()  # default auto
    except Exception:
        pass
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    video_manager.release()

    scenes = []
    for s in scene_list:
        start = s[0].get_seconds()
        end = s[1].get_seconds()
        scenes.append((start, end))
    if not scenes:
        # fallback: full video as one scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        dur = frames / fps if fps else 0
        cap.release()
        scenes = [(0.0, dur)]
    return scenes

# -------------------------
# Visual analysis per scene (brightness, color, shot size rough)
# -------------------------
def analyze_scene_visuals(video_path: str, start_s: float, end_s: float, sample_fps=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    samples = []
    step = max(1, int(fps / sample_fps))
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % step == 0:
            # compute brightness, colorfulness, entropy, aspect ratio center bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            # colorfulness (Hasler & Süsstrunk)
            (B, G, R) = cv2.split(frame.astype("float"))
            rg = np.absolute(R - G)
            yb = np.absolute(0.5*(R + G) - B)
            stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
            meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
            colorfulness = float(stdRoot + (0.3 * meanRoot))
            samples.append({"brightness": brightness, "colorfulness": colorfulness})
        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap.release()
    if not samples:
        return {"brightness": 0.0, "colorfulness": 0.0}
    avg_brightness = float(np.mean([s["brightness"] for s in samples]))
    avg_color = float(np.mean([s["colorfulness"] for s in samples]))
    return {"brightness": avg_brightness, "colorfulness": avg_color}

# -------------------------
# Face emotion analysis (optional DeepFace)
# -------------------------
def analyze_face_emotions(video_path: str, start_s: float, end_s: float, max_faces=5):
    if not have_deepface:
        return {"faces": []}
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    emotions = []
    checked = 0
    while frame_idx <= end_frame and checked < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        # sample every ~1 second
        if (frame_idx - start_frame) % int(fps) == 0:
            try:
                # DeepFace takes BGR as input
                resp = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(resp, list):
                    resp = resp[0]
                emotions.append(resp.get('emotion', {}))
            except Exception:
                pass
            checked += 1
        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap.release()
    # aggregate emotions
    agg = {}
    for e in emotions:
        for k,v in e.items():
            agg[k] = agg.get(k, 0) + v
    # normalize
    total = sum(agg.values()) or 1
    for k in agg:
        agg[k] = agg[k]/len(emotions) if emotions else 0
    return {"faces": emotions, "agg_emotion": agg}

# -------------------------
# Audio analysis (loudness, tempo rough)
# -------------------------
def analyze_audio(video_path: str, start_s: float, end_s: float):
    # extract audio segment using librosa (requires ffmpeg to be available)
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True, offset=start_s, duration=max(0.1, end_s - start_s))
    except Exception:
        return {"rms": 0.0, "tempo": 0.0}
    # RMS (loudness)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    # tempo estimate
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo = 0.0
    return {"rms": rms, "tempo": tempo}

# -------------------------
# (Optional) Whisper speech transcription
# -------------------------
def transcribe_speech_whisper(video_path: str, start_s: float, end_s: float):
    # Optional: if user has whisper installed (openai whisper or whisperx)
    try:
        import whisper  # requires pip install -U openai-whisper
    except Exception:
        return ""
    model = whisper.load_model("small")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpname = tmp.name
    tmp.close()
    # use ffmpeg to extract audio chunk as wav via librosa or ffmpeg-python
    try:
        import ffmpeg
        (
            ffmpeg
            .input(video_path, ss=start_s, t=max(0.1, end_s - start_s))
            .output(tmpname, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception:
        # fallback: librosa write
        try:
            y, sr = librosa.load(video_path, sr=16000, offset=start_s, duration=end_s-start_s)
            sf.write(tmpname, y, sr)
        except Exception:
            return ""
    result = model.transcribe(tmpname, fp16=False)
    try:
        os.unlink(tmpname)
    except Exception:
        pass
    return result.get('text','')

# -------------------------
# LLM explanation (optional)
# -------------------------
def llm_explain(scene_info: dict):
    """
    scene_info: contains keys like visuals, audio, emotions, transcript_excerpt, scene_index
    returns short paragraph explanation
    """
    if openai is None:
        # simple rule-based explanation
        vis = scene_info.get('visuals', {})
        audio = scene_info.get('audio', {})
        em = scene_info.get('emotions', {}).get('agg_emotion', {})
        explanation = []
        # brightness -> tone
        if vis.get('brightness', 0) < 60:
            explanation.append("Low lighting creates a moody or tense atmosphere.")
        else:
            explanation.append("Bright visuals give an open or neutral mood.")
        # colorfulness
        if vis.get('colorfulness',0) < 15:
            explanation.append("Muted colors suggest seriousness or melancholy.")
        # audio loudness
        if audio.get('rms',0) > 0.03:
            explanation.append("Loud audio indicates intensity or action.")
        if em:
            top = sorted(em.items(), key=lambda x:-x[1])[:2]
            if top:
                explanation.append(f"Faces show emotions like {top[0][0]} which supports the scene's mood.")
        # transcript
        txt = scene_info.get('transcript','').strip()
        if txt:
            explanation.append("Dialog suggests: " + (txt[:120] + ("..." if len(txt)>120 else "")))
        return " ".join(explanation)
    # else call openai
    prompt = "You are a film analysis assistant. Given the facts about a scene, produce a concise cinematic explanation (1-2 sentences) of the scene's emotional purpose and cinematic techniques. Facts:\n"
    facts = []
    for k,v in scene_info.items():
        facts.append(f"{k}: {v}")
    prompt += "\n".join(facts)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=120,
            temperature=0.2
        )
        out = resp['choices'][0]['message']['content'].strip()
        return out
    except Exception:
        return ""

# -------------------------
# High-level pipeline
# -------------------------
def analyze_video_cinematic(video_path: str, max_scenes=10, sample_fps=1.0, use_whisper=False, use_llm=False):
    ensure_ffmpeg()
    scenes = detect_scenes(video_path)
    results = []
    for idx, (s,e) in enumerate(scenes[:max_scenes]):
        vis = analyze_scene_visuals(video_path, s, e, sample_fps=sample_fps)
        audio = analyze_audio(video_path, s, e)
        emotions = analyze_face_emotions(video_path, s, e) if have_deepface else {"faces":[], "agg_emotion":{}}
        transcript = ""
        if use_whisper:
            transcript = transcribe_speech_whisper(video_path, s, e)
        scene_info = {
            "scene_index": idx+1,
            "start": s,
            "end": e,
            "visuals": vis,
            "audio": audio,
            "emotions": emotions,
            "transcript": transcript
        }
        scene_info['explanation'] = llm_explain(scene_info) if use_llm or openai else llm_explain(scene_info)
        results.append(scene_info)
    return results

# -------------------------
# TTS helper: join explanations to audio bytes
# -------------------------
def explanations_to_tts(explanations: List[str], lang='en'):
    text = " ".join(explanations)
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        name = tmp.name
        tmp.close()
        tts.save(name)
        with open(name,'rb') as f:
            data = f.read()
        try: os.unlink(name)
        except: pass
        return data
    except Exception:
        return None
