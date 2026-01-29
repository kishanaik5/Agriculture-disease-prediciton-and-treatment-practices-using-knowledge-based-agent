from abc import ABCMeta, abstractmethod

import httpx
from async_lru import alru_cache
from pydantic import BaseModel


class BaseService:
    pass


class GenericService[ModelType: BaseModel](BaseService, metaclass=ABCMeta):
    BASE_URL: str
    HEADERS: dict[str, str]
    TIMEOUT: int = 10

    @abstractmethod
    async def _headers(self, **kwargs) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    async def model2dict(cls, model: ModelType) -> dict:
        raise NotImplementedError

    @classmethod
    @alru_cache
    async def HEADERS(cls, **kwargs) -> dict[str, str]:
        return await cls._headers(**kwargs)

    @classmethod
    async def create(cls, data: ModelType, *, URI: str = None) -> ModelType:
        URI = URI if URI is not None else ""
        async with httpx.AsyncClient() as client:
            resp = await client.post(cls.BASE_URL + URI, headers=await cls.HEADERS(), json=cls.model2dict(data), timeout=cls.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    async def fetch(cls, uid: str, *, URI: str = None) -> ModelType:
        URI = URI if URI is not None else ""
        async with httpx.AsyncClient() as client:
            resp = await client.get(cls.BASE_URL + URI + uid, headers=await cls.HEADERS(), timeout=cls.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    async def update(cls, uid: str, updates: dict, *, URI: str = None) -> ModelType:
        URI = URI if URI is not None else ""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(cls.BASE_URL + URI + uid, headers=await cls.HEADERS(), json=updates,
                                      timeout=cls.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    async def delete(cls, uid: str, *, URI: str = None):
        URI = URI if URI is not None else ""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(cls.BASE_URL + URI + uid, headers=await cls.HEADERS(), timeout=cls.TIMEOUT)
        resp.raise_for_status()
        return resp.json()
