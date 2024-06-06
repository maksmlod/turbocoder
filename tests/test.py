import numpy as np

from turbo.awgn import AWGN
from turbo.turbo_encoder import TurboEncoder
from turbo.turbo_decoder import TurboDecoder

def print_results():
    interleaver = [9, 8, 5, 6, 2, 1, 7, 0, 3, 4]
    encoder = TurboEncoder(interleaver)
    decoder = TurboDecoder(interleaver)

    channel = AWGN(-49)

    input_vector = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    encoded_vector = encoder.execute(input_vector)

    channel_vector = list(map(float, encoded_vector))
    channel_vector = channel.convert_to_symbols(channel_vector)

    channel_vector = channel.execute(channel_vector)

    decoded_vector = decoder.execute(channel_vector)
    decoded_vector = [int(b > 0.0) for b in decoded_vector]

    print("")
    print("--test_turbo_decoder--")
    print("input_vector = {}".format(input_vector))
    print("encoded_vector = {}".format(encoded_vector))
    print("decoded_vector = {}".format(decoded_vector))
'''
    print("Oczekiwany wynik kodowania:")
    print(np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))
    print("Wynik kodowania:")
    print(encoded_output)

    print("\n")
    print("Oczekiwany wynik dekodowania:")
    print(vector)
    print("Wynik dekodowania:")
    print(decoded_output)
'''

if __name__ == '__main__':
    print_results()

