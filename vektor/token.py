import binascii
import re


# TODO
def subword(text, split_length=5):
    ...


# TODO
def word(text, split_length=5):
    ...


# Define a function to tokenize a string into words and punctuation as separate tokens
def tokenize(text):
    # Use regex to match words or punctuation as separate tokens
    tokens = re.findall(
        r"\w+|[^\w\s]", text, re.UNICODE
    )  # This would be moved to `word`
    # use `word`
    # use `subword`
    return tokens


def encode_token_to_bytes(token):
    # Encode the token to its byte representation and convert to an integer
    return int(binascii.hexlify(token.encode()), 16)


def decode_bytes_to_token(encoded_token):
    # Convert the integer back to bytes and then decode to a string
    hex_str = format(encoded_token, "x")  # Convert to hex string
    # Ensure the hex string has an even number of characters for unhexlify
    hex_str = "0" * (len(hex_str) % 2) + hex_str
    return binascii.unhexlify(hex_str).decode()


if __name__ == "__main__":
    # Testing the encoding and decoding
    text = "I am a model."
    tokens = tokenize(text)
    encoded_tokens = [encode_token_to_bytes(token) for token in tokens]
    decoded_tokens = [decode_bytes_to_token(token) for token in encoded_tokens]

    print(f"Original text: {text}")
    print(f"Original tokens: {tokens}")
    print(f"Encoded token: {encoded_tokens}")
    print(f"Decoded token: {decoded_tokens}")
