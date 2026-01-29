import base64
import os
from abc import ABCMeta, abstractmethod

import sqlalchemy as db
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy.ext.asyncio import AsyncEngine

from .base import GenericManager, BaseSchema


class BaseEncryptionSchema(BaseSchema):
    __abstract__ = True

    key = db.Column(db.String)


class BaseEncryptionManager[SchemaType: BaseEncryptionSchema](GenericManager[SchemaType], metaclass=ABCMeta):
    UID_SIZE = 8

    @staticmethod
    def int2str(value: int) -> str:
        byte_representation = value.to_bytes((value.bit_length() + 7) // 8, byteorder='big') or b'\0'
        return base64.urlsafe_b64encode(byte_representation).decode('utf-8')

    @staticmethod
    def str2int(value: str) -> int:
        byte_representation = base64.urlsafe_b64decode(value.encode('utf-8'))
        return int.from_bytes(byte_representation, byteorder='big')

    @staticmethod
    def bytes2str(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode('utf-8')

    @staticmethod
    def str2bytes(value: str) -> bytes:
        return base64.urlsafe_b64decode(value.encode('utf-8'))

    @abstractmethod
    async def generate_key(self, key_size=32) -> str:
        raise NotImplementedError

    @abstractmethod
    async def encrypt(self, plaintext: str, uid: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def decrypt(self, ciphertext: str, uid: str) -> str:
        raise NotImplementedError


class AES256EncryptionSchema(BaseEncryptionSchema):
    __tablename__ = "aes256_encryption_keys"


class AES256EncryptionManager(BaseEncryptionManager[AES256EncryptionSchema]):
    def _derive_key(self, salt: bytes, size: int) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=size,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)

    def __init__(self, engine: AsyncEngine, *, password: str):
        super().__init__(engine)
        self.password = password.encode()
        self.backend = default_backend()

    async def generate_key(self, key_size=32) -> str:
        salt = os.urandom(key_size)
        key = self._derive_key(salt, key_size)
        return self.bytes2str(salt + key)

    async def encrypt(self, plaintext: str, uid: str) -> str:
        secret = (await self.fetch(uid)).key
        key = self.str2bytes(secret)
        salt, key = key[:(m := len(key) // 2)], key[m:]
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        return self.bytes2str(iv + ciphertext)

    async def decrypt(self, ciphertext: str, uid: str) -> str:
        secret = (await self.fetch(uid)).key
        key = self.str2bytes(secret)
        salt, key = key[:(m := len(key) // 2)], key[m:]
        ciphertext = self.str2bytes(ciphertext)
        iv, ciphertext = ciphertext[:16], ciphertext[16:]
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()


__all__ = [
    "BaseEncryptionSchema", "BaseEncryptionManager",
    "AES256EncryptionSchema", "AES256EncryptionManager"
]
