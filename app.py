from flask import Flask, request, render_template
import numpy as np
from turbo.awgn import AWGN
from turbo.turbo_encoder import TurboEncoder
from turbo.turbo_decoder import TurboDecoder
import random

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_vector = request.form['input_text']
        input_vector = [int(bit) for bit in input_vector if bit.isdigit()]
        interleaver = random.sample(range(len(input_vector)), len(input_vector))
        encoded_output, decoded_output = process_data(input_vector, interleaver)
        return render_template('index.html', input_vector=input_vector, interleaver=interleaver, encoded_output=encoded_output, decoded_output=decoded_output)
    return render_template('index.html')

def process_data(input_vector, interleaver):
    encoder = TurboEncoder(interleaver)
    decoder = TurboDecoder(interleaver)
    channel = AWGN(0)

    encoded_vector = encoder.execute(input_vector)

    channel_vector = list(map(float, encoded_vector))
    channel_vector = channel.convert_to_symbols(channel_vector)
    channel_vector = channel.execute(channel_vector)

    decoded_vector = decoder.execute(channel_vector)
    decoded_vector = [int(b > 0.0) for b in decoded_vector]

    return encoded_vector, decoded_vector

if __name__ == '__main__':
    app.run(debug=True)
