from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
from turbo.awgn import AWGN
from turbo.turbo_encoder import TurboEncoder
from turbo.turbo_decoder import TurboDecoder
import random
import io

app = Flask(__name__)
awgn_level = 0  # Default AWGN level


@app.route('/', methods=['GET', 'POST'])
def index():
    global awgn_level
    if request.method == 'POST':
        input_vector = request.form['input_text']
        input_vector = [int(bit) for bit in input_vector if bit.isdigit()]
        awgn_level = int(request.form['awgn_level'])
        generate_plot_flag = 'generate_plot' in request.form
        interleaver = random.sample(range(len(input_vector)), len(input_vector))
        encoded_output, decoded_output, channel_vector_before, channel_vector_after = process_data2(input_vector, interleaver, awgn_level)
        error_rate = calculate_error_rate(input_vector, decoded_output)

        if generate_plot_flag:
            generate_plot(input_vector)

        return render_template('index.html', input_vector=input_vector, interleaver=interleaver,
                               awgn_level=awgn_level, encoded_output=encoded_output, decoded_output=decoded_output,
                               error_rate=error_rate, generate_plot_flag=generate_plot_flag,
                               channel_vector_before=channel_vector_before, channel_vector_after=channel_vector_after)
    return render_template('index.html', awgn_level=awgn_level)


def process_data(input_vector, interleaver, awgn_level):
    encoder = TurboEncoder(interleaver)
    decoder = TurboDecoder(interleaver)
    if awgn_level == 0: awgn_level = -1000
    channel = AWGN(awgn_level)

    encoded_vector = encoder.execute(input_vector)

    channel_vector = list(map(float, encoded_vector))
    channel_vector = channel.convert_to_symbols(channel_vector)
    channel_vector = channel.execute(channel_vector)

    decoded_vector = decoder.execute(channel_vector)
    decoded_vector = [int(b > 0.0) for b in decoded_vector]

    decoded_vector = decoded_vector[:-2]

    return encoded_vector, decoded_vector


def process_data2(input_vector, interleaver, awgn_level):
    encoder = TurboEncoder(interleaver)
    decoder = TurboDecoder(interleaver)
    channel1 = AWGN(-1000)
    if awgn_level == 0: awgn_level = -1000
    channel2 = AWGN(awgn_level)

    encoded_vector = encoder.execute(input_vector)
    encoded_vector2 = encoded_vector

    channel_vector_before = list(map(float, encoded_vector))
    channel_vector_before = channel1.convert_to_symbols(channel_vector_before)
    channel_vector_before = channel1.execute(channel_vector_before)


    channel_vector_after = list(map(float, encoded_vector2))
    channel_vector_after = channel2.convert_to_symbols(channel_vector_after)
    channel_vector_after = channel2.execute(channel_vector_after)

    decoded_vector = decoder.execute(channel_vector_after)
    decoded_vector = [int(b > 0.0) for b in decoded_vector]

    decoded_vector = decoded_vector[:-2]

    return encoded_vector, decoded_vector, channel_vector_before, channel_vector_after


def calculate_error_rate(input_vector, decoded_vector):
    error_count = sum(1 for x, y in zip(input_vector, decoded_vector) if x != y)
    total_bits = len(input_vector)
    print(error_count)
    error_rate = (error_count / total_bits) * 100
    return error_rate


def generate_plot(input_vector):
    awgn_values = range(0, 100, 2)
    error_rates = []

    for awgn in awgn_values:
        total_error_rate = 0
        interleaver = random.sample(range(len(input_vector)), len(input_vector))
        for _ in range(40):
            _, decoded_output = process_data(input_vector, interleaver, awgn)
            total_error_rate += calculate_error_rate(input_vector, decoded_output)
        average_error_rate = total_error_rate / 50
        error_rates.append(average_error_rate)

    plt.figure(figsize=(6, 4))
    plt.plot(awgn_values, error_rates, marker='o', linestyle='-')
    plt.title('Average Error Rate(AWGN)')
    plt.xlabel('AWGN [dB]')
    plt.ylabel('Average Error Rate [%]')
    plt.grid(True)
    plt.savefig('static/error_rate_plot.png')
    plt.close()



@app.route('/plot')
def plot():
    return send_file('static/error_rate_plot.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
