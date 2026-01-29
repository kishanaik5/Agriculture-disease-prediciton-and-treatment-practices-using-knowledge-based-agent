from enum import Enum
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from SharedBackend.managers import BaseSchema


class BaseErrorCode(Enum):
    def __init__(self, code: int, message: str):
        self._value_ = code
        self.message = message

    @property
    def code(self):
        return self.value

    def as_dict(self, extras, **kwargs):
        return {
            "code": self.code,
            "message": self.message.format(**kwargs),
            "name": self.name,
            "extras": extras
        }


class DBErrorCode(BaseErrorCode):
    DB_UNKNOWN_ERROR = (1000, "unknown error")
    DB_NOT_FOUND = (1001, "record not found for {__tablename__}")
    DB_MULTIPLE_FOUND = (1002, "multiple records found for {__tablename__}")
    DB_INTEGRITY_ERROR = (1003, "Integrity Error: {__tablename__}")


class EnumException(HTTPException):
    def __init__(self, status_code, error_enum: BaseErrorCode, headers=None, extras=None, err_kwargs=None):
        if err_kwargs is None:
            err_kwargs = err_kwargs or {}
        super().__init__(status_code, detail=error_enum.as_dict(extras, **err_kwargs), headers=headers)


class GenericSchemaException(EnumException):
    def __init__(self, status_code, error_enum: BaseErrorCode, schema: "BaseSchema", headers=None, extras=None):
        super().__init__(status_code, error_enum, headers, err_kwargs={"__tablename__": schema.__tablename__},
                         extras=extras)


__all__ = [
    "BaseErrorCode",
    "DBErrorCode",
    "EnumException",
    "GenericSchemaException",
]
