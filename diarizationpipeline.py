
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
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
import smtplib
import pandas as pd

from dotenv import load_dotenv
load_dotenv(".env")
EMAIL=os.getenv("EMAIL")
PASSWORD=os.getenv("PASSWORD")
HF_TOKEN=os.getenv("HF_TOKEN")


home_path='.'
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
    # Attach each file in the folder
    folder_path=f'{home_path}/results'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {filename}")
                msg.attach(part)
      
      # with open(f'{home_path}/results/', "rb") as f:
      #     img_data = f.read()
      # image = MIMEImage(img_data, name="trans.txt",_subtype="txt")
      # msg.attach(image)

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
  return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=HF_TOKEN)

pipeline_dia = load_pipeline_dia()



def final(link,file_name,email,type):
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
          audio_file_name=f'{home_path}/{x[0]}.wav'
          a.export(audio_file_name, format="wav")
          file1.write(speaker+": "+transcribe_speech(audio_file_name)+"\n")
          os.remove(audio_file_name)
          
          # os.remove(f'{home_path}/download')

  except Exception as e:
    print(e)

  else:
    if type=='folder':
       pass
    elif type=='single':
      send_email(email)
    print("ALL GOOD!")

#https://www.youtube.com/watch?v=nBpPe9UweWs
file_name=f'trans.mp4'



st.title("Youtube Diarizaiton - ver1")

with st.form(key='input_form'):
  link = st.text_input("Enter the youtube link here")
  email = st.text_input("Enter your email address")
  file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])
  submit_button = st.form_submit_button(label='Submit')

  # When the form is submitted, call the process_input function
  if submit_button:
    # Validate that both fields are not empty
    if email and link:
      threading.Thread(target=final, args=(link,file_name,email,'single')).start()
      st.success("File uploaded successfully! You will receive an email with the results shortly.")
    elif email and file:
      df = pd.read_excel(file)
      # Check if the 'LINKS' column exists
      if 'LINKS' in df.columns:
        links = df['LINKS'].tolist()
        st.write("Links from the Excel file:")
        threads=[]
        for i, link in enumerate(links, 1):
          thread=threading.Thread(target=final, args=(link,f'{i}.mp4',email,'folder'))
          thread.start()
          threads.append(thread)
        st.success("File uploaded successfully! You will receive an email with the results shortly.")
        for thread in threads:
          thread.join()
        
        send_email(email)
          

          
      else:
        st.write("The uploaded file does not contain a 'LINKS' column.")
    else:
      st.error("not all fields present")

