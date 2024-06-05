
import streamlit as st
import moviepy.editor as mp
from pydub import AudioSegment
import os
from pytube import YouTube
from pytube.innertube import _default_clients
from pyannote.audio import Pipeline
from transformers import pipeline
import threading
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import smtplib

from dotenv import load_dotenv
load_dotenv(".env")
EMAIL=os.getenv("EMAIL")
PASSWORD=os.getenv("PASSWORD")


home_path=''
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def Download(link,file_path,file_name):
  print("STARTING DOWNLOAD")
  youtubeObject = YouTube(link)
  youtubeObject = youtubeObject.streams.get_highest_resolution()
  try:
    youtubeObject.download(output_path=file_path,filename=file_name)
  except:
    print("An error has occurred")
  else:
    print("Download is completed successfully")

# Function to send email
def send_email(receiver_email):
  print("Sending email...")
  msg = MIMEMultipart()
  msg['Subject'] = 'Diarization Report'
  msg['From'] = EMAIL
  msg['To'] = receiver_email
  global INPUT_PATH
  try:
      
      with open(f'{home_path}/results/trans.txt', "rb") as f:
          img_data = f.read()
      image = MIMEImage(img_data, name="trans.txt",_subtype="txt")
      msg.attach(image)

  except Exception as e:
      print("Error attaching image:", e)
  text = MIMEText("Report attached.")
  msg.attach(text)
  s = smtplib.SMTP('smtp.gmail.com', 587)
  s.ehlo()
  s.starttls()
  s.ehlo()
  s.login( EMAIL, PASSWORD)
  s.sendmail( EMAIL, receiver_email, msg.as_string())
  s.quit()
  print("Mail Sent")



@st.cache_resource
def load_asr():
  return pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en")

asr = load_asr()
def transcribe_speech(filepath):
  if filepath is None:
    return "No Audio"
  output = asr(filepath)
  return output["text"]


@st.cache_resource
def load_pipeline_dia():
  return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")

pipeline_dia = load_pipeline_dia()

"""### Final Function"""

def final(link,file_name,email):
  try:
    Download(link,'downloads',file_name)

    video = mp.VideoFileClip(f'{home_path}/downloads/{file_name}')

    # Extract the audio from the video
    print(".wav File Generation Started")
    audio_file = video.audio
    audio_file.write_audiofile(f'{home_path}/downloads/{file_name}.wav')
    print(".wav File Generated")
    print("Diarization Started")
    diarization = pipeline_dia(f'{home_path}/downloads/{file_name}.wav')
    print("Diarization Completed")
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    try:
      os.mkdir('results')
    except:
      pass

    print("Preparing result")
    x=file_name.split('.')
    with open(f'{home_path}/results/{x[0]}.txt', "a") as file1:
      newAudio = AudioSegment.from_wav(f'{home_path}/downloads/{file_name}.wav')
      for segment, _, speaker in diarization.itertracks(yield_label=True):
          t1 = float(segment.start) * 1000 # works in milliseconds
          t2 = float(segment.end) * 1000
          a = newAudio[t1:t2]
          audio_file_name=f'{home_path}/WASTE.wav'
          a.export(audio_file_name, format="wav")
          file1.write(speaker+": "+transcribe_speech(audio_file_name)+"\n")
          os.remove(audio_file_name)
          os.remove(f'{home_path}//download')

  except Exception as e:
    print(e)

  else:
    send_email(email)
    return("ALL GOOD!")

#https://www.youtube.com/watch?v=nBpPe9UweWs
file_name=f'{home_path}/trans.mp4'



st.title("Youtube Diarizaiton - ver1")

# Apply some CSS to style the app


# Email input
link = st.text_input("Enter the youtube link here")
email = st.text_input("Enter your email address")
if link and email :
  # final(link,file_name)
  threading.Thread(target=final, args=(link,file_name,email)).start()
  st.success("File uploaded successfully! You will receive an email with the results shortly.")
  link=None
