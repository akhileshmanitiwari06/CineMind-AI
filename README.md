# CineMind-AI
AI that explains the psychology, emotion, and cinematography behind movie scenes.

🎯 Goal

Understand camera angles, emotions, dialogue tones → explain “why this scene feels powerful.”

💡 Use Case

Film students, critics, or YouTubers (automatic scene breakdowns).

🧩 Architecture

Video Input → Scene Splitter

Vision Analysis – emotion, brightness, camera angle.

Audio Analysis – background tone, dialogue mood.

LLM Commentary – generate analysis (“This low-angle shot shows dominance…”).

⚙️ Tech Stack

OpenCV + PySceneDetect (scene segmentation)

FER / DeepFace (emotion recognition)

Librosa (audio mood analysis)

GPT / Claude / Gemini (scene explanation)

Streamlit UI

🚀 Steps

Split video into scenes.

Detect faces & emotions per frame.

Analyze lighting + sound.

Prompt LLM:

“Analyze the emotional and cinematic meaning of this scene using visual and audio cues.”

Display summary with emotion graphs + text analysis.
