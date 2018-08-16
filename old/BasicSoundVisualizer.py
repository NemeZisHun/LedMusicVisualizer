"""
Notebook for streaming data from a microphone in realtime

audio is captured using pyaudio
then converted from binary data to ints using struct
then displayed using matplotlib

if you don't have pyaudio, then run

note: with 2048 samples per chunk, I'm getting 20FPS
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.fftpack import fft
from tkinter import TclError

# use this backend to display in separate Tk window
#%matplotlib tk

# constants
Chunk = 1024 * 2  # samples per frame
Format = pyaudio.paInt16     # audio format (bytes per sample?)
Channels = 1                 # single channel for microphone
Rate = 44100                 # samples per second
IntSize = 256**2

# create matplotlib figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=Format,
    channels=Channels,
    rate=Rate,
    input=True,
    output=True,
    frames_per_buffer=Chunk
)

# initial variable for plotting
x = np.arange(0, Chunk)       # samples (waveform)
xf = np.linspace(0, Rate, Chunk)     # frequencies (spectrum)

# create a line object with random data
line, = ax1.plot(x, np.random.randint(0,IntSize-1,Chunk,dtype='uint16'), '-', lw=1)

# create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(Chunk)*IntSize, '-', lw=1)
#line_fft, = ax2.plot(xf, np.random.rand(Chunk)*IntSize, '-', lw=1)

# basic formatting for the axes
ax1.set_title('Audio in Time-domain')
ax1.set_xlabel('samples')
ax1.set_ylabel('values')
ax1.set_ylim(0, IntSize-1)
ax1.set_xlim(0, Chunk)

ax2.set_title('Audio in Freq.-domain')
ax2.set_xlabel('frequency')
ax2.set_ylabel('amplitude')
ax2.set_xlim(20,Rate/2)

plt.tight_layout()
plt.setp(ax1, xticks=[0, Chunk/4, Chunk/2, Chunk/4*3, Chunk], yticks=[0, IntSize/4, IntSize/2, IntSize/4*3, IntSize ])

# show the plot with grid
plt.grid()
plt.show(block=False)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:

    # binary data
    data = stream.read(Chunk)

    # convert data to int16, make np array, then offset it by IntSize/2
    data_int = np.fromstring(data, dtype=np.uint16) + np.uint16(IntSize/2) # uncomment last part for more intuitive waveform, this affects the FFT peaks significantly!
    data_fft=np.abs(np.fft.fft(data_int) * 2 / Chunk) *5 #arbitrary scaling to better fill the FFT

    line.set_ydata(data_int)
    line_fft.set_ydata(data_fft)

    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1

    except TclError:

        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)

        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break



