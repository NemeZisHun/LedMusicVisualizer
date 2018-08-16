import numpy as np
import matplotlib.pyplot as plt
import struct
import pyaudio
import time
import scipy

#%matplotlib tk

Chunk = 1024 * 4
Format = pyaudio.paInt16
Channels = 1
Rate = 44100

p = pyaudio.PyAudio()
stream = p.open(
    format = Format,
    channels = Channels,
    rate = Rate,
    input = True,
    output = True,
    frames_per_buffer=Chunk
)
'''
data = stream.read(Chunk)
data_int = np.fromstring(data, dtype=np.uint8)[::2] + 127
fig, ax = plt.subplots()
ax.plot(data_int)
plt.show()
'''

data = stream.read(Chunk)
data_int = np.fromstring(data, dtype=np.uint8)[::2] + 127

fig, ax = plt.subplots()
x = np.arange(0, 2*Chunk, 2)
line, = ax.plot(data_int)
plt.show(block=False)
ax.set_ylim(0,255)
ax.set_xlim(0,Chunk)

#''
while True:
    data = stream.read(Chunk)
    #data_int = np.array(struct.unpack(str(2 * Chunk) + 'B', data)[::2], dtype = 'B') +127
    data_int = np.fromstring(data, dtype=np.uint8)[::2] + 127
    line.set_ydata(data_int)
    fig.canvas.draw()
    fig.canvas.flush_events()
#'''


