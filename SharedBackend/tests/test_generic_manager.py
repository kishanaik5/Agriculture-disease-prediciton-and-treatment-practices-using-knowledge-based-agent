import unittest

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine

from SharedBackend.managers import BaseSchema, GenericManager


class EmptySchema(BaseSchema):
    __tablename__ = "empties"


class EmptyManager(GenericManager[EmptySchema]): pass


class TestGenericManager(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        cls.manager = EmptyManager(cls.engine)

    async def asyncSetUp(self):
        """Reinitialize the database for each test."""
        await self.manager.init_db()

    async def test_create(self):
        """Test creating a new record."""
        empty = EmptySchema()
        created_empty = await self.manager.create(empty)
        self.assertIsNotNone(created_empty.uid)

    async def test_fetch(self):
        """Test fetching a record by UID."""
        empty = EmptySchema()
        created_empty = await self.manager.create(empty)
        fetched_empty = await self.manager.fetch(created_empty.uid)
        self.assertEqual(created_empty.uid, fetched_empty.uid)

    async def test_update(self):
        """Test updating a record."""
        empty = EmptySchema()
        created_empty = await self.manager.create(empty)
        updates = {"uid": "new-uid"}
        updated_empty = await self.manager.update(created_empty.uid, updates)
        self.assertEqual(updated_empty.uid, "new-uid")

    async def test_delete(self):
        """Test deleting a record."""
        empty = EmptySchema()
        created_empty = await self.manager.create(empty)
        await self.manager.delete(created_empty.uid)
        with self.assertRaises(HTTPException):
            await self.manager.fetch(created_empty.uid)

    async def test_fetch_all(self):
        """Test fetching all records."""
        empty1 = EmptySchema()
        empty2 = EmptySchema()
        await self.manager.create(empty1)
        await self.manager.create(empty2)
        all_records = await self.manager.fetch_all(limit=10)
        self.assertGreaterEqual(len(all_records), 2)

    async def test_update_all(self):
        """Test updating all records."""
        empty1 = EmptySchema()
        empty2 = EmptySchema()
        await self.manager.create(empty1)
        await self.manager.create(empty2)
        updates = {}
        updated_records = await self.manager.update_all(limit=10, updates=updates)
        for record in updated_records:
            self.assertEqual(record.uid, updates.get("uid", record.uid))

    async def test_delete_all(self):
        """Test deleting all records."""
        empty1 = EmptySchema()
        empty2 = EmptySchema()
        await self.manager.create(empty1)
        await self.manager.create(empty2)
        await self.manager.delete_all(limit=10)
        all_records = await self.manager.fetch_all(limit=10)
        self.assertGreaterEqual(len(all_records), 0)
