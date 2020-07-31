import os

# Specify the paths
input_path = os.path.join('.', 'frames', 'frame_%07d.png')
output_path = os.path.join('.', 'simulation.mp4')

# Set the encoder options
# See the ffmpeg docs at https://trac.ffmpeg.org/wiki/Encode/H.264
options = ['-f image2',
           '-r 60',
           '-i ' + input_path,
           '-c:v libx264',
           '-preset slow',  # Choose encode speed from e.g. [veryfast, fast, medium, slow, veryslow]
           '-tune grain',  # Choose encode tune from e.g. [film, animation, grain]
           '-crf 15',  # Choose value between 0 and 50, 0 is lossless, 50 is max compression, typical range is 15-30
           '-y ' + output_path]

# Perform the encoding
os.system('ffmpeg ' + ' '.join(options))
