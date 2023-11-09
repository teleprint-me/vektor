"""
vektor/tokenizer.py
"""
import binascii
import re


class Tokenizer:
    # Compile the regular expression for tokenization
    TOKEN_REGEX = re.compile(r"\w+|[^\w\s]|'\w+", re.UNICODE)

    @staticmethod
    def tokenize(text):
        """Tokenize the input text into a list of tokens."""
        return Tokenizer.TOKEN_REGEX.findall(text)

    @staticmethod
    def encode_to_bytes(token):
        """Encode the token to its byte representation."""
        try:
            return int(binascii.hexlify(token.encode()), 16)
        except UnicodeEncodeError as e:
            error_msg = f"Encoding failed for token: {token}"
            # Consider using logging.error(error_msg)
            raise UnicodeEncodeError(error_msg) from e

    @staticmethod
    def decode_from_bytes(encoded_token):
        """Decode the encoded token back to its string representation."""
        try:
            # Ensures that hex_str is padded to even length
            hex_str = format(encoded_token, "x").zfill(2)
            return binascii.unhexlify(hex_str).decode()
        except (binascii.Error, ValueError) as e:
            error_msg = f"Decoding failed for encoded token: {encoded_token}"
            # Consider using logging.error(error_msg)
            raise ValueError(error_msg) from e


if __name__ == "__main__":
    # Testing the encoding and decoding
    text = "こんにちは！ これは単なるテストです。"
    tokens = Tokenizer.tokenize(text)
    encoded_tokens = [Tokenizer.encode_to_bytes(token) for token in tokens]
    decoded_tokens = [Tokenizer.decode_from_bytes(token) for token in encoded_tokens]

    print(f"Original text: {text}")
    print(f"Original tokens: {tokens}")
    print(f"Encoded tokens: {encoded_tokens}")
    print(f"Decoded tokens: {decoded_tokens}")
