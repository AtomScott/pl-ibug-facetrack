import os
import subprocess

class FFMPEGFrames:
    def __init__(self, output):
        self.output = output

    def extract_frames(self, input, fps):
        output = input.split('/')[-1].split('.')[0]

        if not os.path.exists(self.output + output):
            os.makedirs(self.output + output)
        # query = "ffmpeg -i " + input + " -vf fps=" + str(fps) + " " + self.output + output + "/output%06d.png"

        query = "ffmpeg -i " + input + " " + self.output + output + "/output%06d.png"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        s = str(response).encode('utf-8')

    def stitch_images(self, input_left, input_right):
        output = os.path.join(self.output, 'stitched')

        if not os.path.exists(output):
            os.makedirs(output)

        for i, (frame_l , frame_r) in enumerate(zip(input_left, input_right)):

            query = f"ffmpeg -i {frame_l} -i {frame_r} -y -filter_complex hstack {output}/{i+1:06}.jpg"
            response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
            s = str(response).encode('utf-8')
