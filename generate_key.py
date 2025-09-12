"""
Script para generar una clave Fernet válida para IOL_TOKENS_KEY.
Uso:
    python generate_key.py
"""

from cryptography.fernet import Fernet

def main():
    key = Fernet.generate_key().decode()
    print("\nTu nueva clave IOL_TOKENS_KEY es:\n")
    print(key)
    print("\n📌 Copiala en tu archivo .env como:\n")
    print(f'IOL_TOKENS_KEY="{key}"\n')

if __name__ == "__main__":
    main()