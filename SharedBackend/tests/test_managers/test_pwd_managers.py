import unittest

from sqlalchemy.ext.asyncio import create_async_engine

from SharedBackend.managers.pwdManager import UserPasswordManager


class TestAPIKeyManager(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        cls.manager = UserPasswordManager(cls.engine)

    async def asyncSetUp(self):
        await self.manager.init_db()

    async def test_register_password(self):
        username = "test_username"
        password = "test_password"
        await self.manager.register_password(password, username=username)
        record = await self.manager.verify_password(password, username=username)
        self.assertIsNotNone(record)
        self.assertEqual(record.username, username)
        await self.manager.revoke_password(record.uid)

    async def test_revoke_password(self):
        username = "test_username"
        password = "test_password"
        record = await self.manager.register_password(password, username=username)
        await self.manager.revoke_password(record.uid)
        record = await self.manager.verify_password(password, username=username)
        self.assertIsNone(record)


if __name__ == '__main__':
    unittest.main()
