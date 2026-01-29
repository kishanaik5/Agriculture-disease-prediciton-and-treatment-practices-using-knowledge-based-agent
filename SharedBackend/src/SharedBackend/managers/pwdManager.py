from abc import ABCMeta

import sqlalchemy as db

from .base import GenericManager, BaseSchema


class BasePassSchema(BaseSchema):
    __abstract__ = True

    password_hash = db.Column(db.String)

    @property
    def password(self) -> str:
        raise AttributeError("`password` is a write-only property")

    @password.setter
    def password(self, value: str):
        self.password_hash = self._hash_str(value)

    @property
    def password_salt(self) -> str:
        return self._split_hash_str(self.password_hash)[0]

    def __password_eq__(self, password: str) -> bool:
        return self.password_hash == self._hash_str(password, salt=self.password_salt)


class BasePassManager[SchemaType: BasePassSchema](GenericManager[SchemaType], metaclass=ABCMeta):
    pass


__all__ = [
    "BasePassSchema", "BasePassManager"
]
