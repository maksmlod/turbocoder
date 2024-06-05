import numpy as np
from turbo.turbo_encoder import TurboEncoder
from turbo.turbo_decoder import TurboDecoder

def print_results():
    interleaver = [2, 0, 3, 1]
    encoder = TurboEncoder(interleaver)
    decoder = TurboDecoder(interleaver)

    vector = np.array([1, 0, 1, 0])
    encoded_output = encoder.execute(vector)
    decoded_output = decoder.execute(encoded_output)

    print("Oczekiwany wynik kodowania:")
    print(np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))
    print("Wynik kodowania:")
    print(encoded_output)

    print("\n")
    print("Oczekiwany wynik dekodowania:")
    print(vector)
    print("Wynik dekodowania:")
    print(decoded_output)

if __name__ == '__main__':
    print_results()

