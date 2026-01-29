import unittest

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine

from SharedBackend.managers import *


class TestAES256EncryptionManager(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        cls.manager = AES256EncryptionManager(cls.engine, password="12345678-unsafe-password")

    async def asyncSetUp(self):
        await self.manager.init_db()

    async def test_generate_key(self):
        key_size = 32
        key = await self.manager.generate_key(key_size=key_size)
        self.assertEqual(len(key), key_size * 2.75)

    async def test_register_key(self):
        key = await self.manager.generate_key()
        record = await self.manager.create(AES256EncryptionSchema(key=key))
        self.assertEqual(record.key, key)

    async def test_revoke_key(self):
        key = await self.manager.generate_key()
        record = await self.manager.create(AES256EncryptionSchema(key=key))
        await self.manager.delete(record.uid)
        with self.assertRaises(HTTPException):
            await self.manager.fetch(record.uid)

    async def test_encrypt_decrypt(self):
        key = await self.manager.generate_key()
        record = await self.manager.create(AES256EncryptionSchema(key=key))
        plaintext = "hello world"
        ciphertext = await self.manager.encrypt(plaintext, record.uid)
        self.assertNotEqual(plaintext, ciphertext)
        self.assertEqual(plaintext, await self.manager.decrypt(ciphertext, record.uid))


if __name__ == '__main__':
    unittest.main()
