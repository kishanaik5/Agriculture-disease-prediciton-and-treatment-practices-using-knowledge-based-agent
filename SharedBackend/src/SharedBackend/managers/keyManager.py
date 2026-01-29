import secrets
from abc import ABCMeta, abstractmethod
import sqlalchemy as db
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import relationship

from .base import GenericManager, BaseSchema

api_key_scopes = db.Table(
    'api_key_scopes',
    BaseSchema.metadata,
    db.Column('api_key_uid', db.String, db.ForeignKey('api_keys.uid'), primary_key=True),
    db.Column('scope_name', db.String, db.ForeignKey('scopes.name'), primary_key=True)
)


class BaseScopeSchema(BaseSchema):
    __tablename__ = "scopes"
    __mapper_args__ = {
        "polymorphic_identity": "base_scope",
        "polymorphic_on": "type",
    }

    name = db.Column(db.String, primary_key=True, unique=True)
    description = db.Column(db.String, nullable=True)
    type = db.Column(db.String, nullable=False)

    api_keys = relationship("BaseKeySchema", secondary=api_key_scopes, back_populates="scopes")


class BaseKeySchema(BaseSchema):
    __tablename__ = "api_keys"
    __mapper_args__ = {
        "polymorphic_identity": "base_key",
        "polymorphic_on": "type",
    }

    key_hash = db.Column(db.String)
    type = db.Column(db.String, nullable=False)

    scopes = relationship("BaseScopeSchema", secondary=api_key_scopes, back_populates="api_keys")

    @property
    def key(self) -> str:
        raise AttributeError("`key` is a write-only property")

    @key.setter
    def key(self, value: str):
        self.key_hash = self._hash_str(value)

    @property
    def key_salt(self) -> str:
        return self._split_hash_str(self.key_hash)[0]

    def __key_eq__(self, key: str) -> bool:
        return self.key_hash == self._hash_str(key, salt=self.key_salt)


class BaseKeyManager[SchemaType: BaseKeySchema](GenericManager[SchemaType], metaclass=ABCMeta):
    KEY_SIZE = 32
    UID_SIZE = 8
    DEFAULT_SCOPES = {"key:auth"}

    @abstractmethod
    async def generate_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def register_key(self, key: str, *scopes: str) -> SchemaType:
        raise NotImplementedError

    @abstractmethod
    async def verify_key(self, key: str) -> set[str] | None:
        raise NotImplementedError


class ApiScopeSchema(BaseScopeSchema):
    __mapper_args__ = {
        "polymorphic_identity": "api_scope",
    }


class ApiKeySchema(BaseKeySchema):
    __mapper_args__ = {
        "polymorphic_identity": "api_key",
    }


class ApiScopeManager(GenericManager[ApiScopeSchema]): pass


class ApiKeyManager(BaseKeyManager[ApiKeySchema]):
    def __init__(self, engine: AsyncEngine):
        super().__init__(engine)
        self.scope_manager = ApiScopeManager(engine)

    def generate_key(self) -> str:
        return secrets.token_hex(self.UID_SIZE) + secrets.token_hex(self.KEY_SIZE)

    async def register_key(self, key: str, *scopes: str) -> ApiKeySchema:
        scopes = {*self.DEFAULT_SCOPES, *scopes}
        uid = key[:self.UID_SIZE]
        record = ApiKeySchema(uid=uid)
        record.key = key
        async with self.session_factory() as session:
            for scope_name in scopes:
                scope = await self.scope_manager.create_or_fetch(ApiScopeSchema(name=scope_name))
                record.scopes.append(scope)
            session.add(record)
            await session.commit()
        return record

    async def verify_key(self, key: str) -> set[str] | None:
        uid = key[:self.UID_SIZE]
        try:
            record = await self.fetch(uid, joins=[ApiKeySchema.scopes])
        except HTTPException:
            return None
        if record.__key_eq__(key):
            return {scope.name for scope in record.scopes}


__all__ = [
    "BaseKeySchema", "BaseScopeSchema", "BaseKeyManager",
    "ApiKeySchema", "ApiScopeManager", "ApiKeyManager",
]
