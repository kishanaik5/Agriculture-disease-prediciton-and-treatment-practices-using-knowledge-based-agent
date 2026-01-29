import asyncio
import hashlib
import secrets
import uuid
from collections import defaultdict
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, get_args, Union, ForwardRef, Annotated, Callable, Literal, Any

import sqlalchemy as db
from pydantic import BaseModel, create_model, PlainSerializer, ConfigDict
from pydantic.main import IncEx
from sqlalchemy import event
from sqlalchemy.exc import MissingGreenlet, MultipleResultsFound, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy.orm import QueryableAttribute, load_only, defer, aliased
from sqlalchemy.orm import sessionmaker, DeclarativeBase, joinedload
from sqlalchemy.orm.exc import DetachedInstanceError

from SharedBackend.exc.base import GenericSchemaException, DBErrorCode

NESTED_JOINS = QueryableAttribute | list["NESTED_JOINS"]
NESTED_FILTERS = dict[str, Union["NESTED_FILTERS", list["NESTED_FILTERS"], str, list[str]]]

DecimalAsFloat = Annotated[
    Decimal,
    PlainSerializer(
        lambda x: float(x),
        return_type=float,
        when_used="json"
    )
]


class BaseSchema(DeclarativeBase):
    __abstract__ = True

    __model__: type[BaseModel] = None
    __post_model__: type[BaseModel] = None
    __patch_model__: type[BaseModel] = None
    __put_model__: type[BaseModel] = None

    SQLA_TYPE_MAPPING = {
        db.Integer: int,
        db.INTEGER: int,
        db.String: str,
        db.Float: float,
        db.FLOAT: float,
        db.JSON: dict,
        db.Boolean: bool,
        db.BOOLEAN: bool,
        db.Numeric: DecimalAsFloat,
        db.NUMERIC: DecimalAsFloat,
        db.DateTime: datetime,
        db.DATETIME: datetime,
        db.Date: date,
        db.DATE: date,
        db.BigInteger: int,
        db.BIGINT: int,
    }
    SALT_SIZE = 16

    uid = db.Column(db.String, primary_key=True, default=lambda: f"{uuid.uuid4()}",
                    info={"readonly": True, "flags": {"post:exclude", "patch:exclude"}})
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(),
                           info={"readonly": True, "flags": {"post:exclude", "patch:exclude"}})
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=db.func.now(),
                           info={"readonly": True, "flags": {"post:exclude", "patch:exclude"}})
    deleted_at = db.Column(db.DateTime(timezone=True), nullable=True,
                           info={"readonly": True, "flags": {"post:exclude", "patch:exclude"}})

    @classmethod
    def get_field_type(cls, col):
        if isinstance(col.type, db.Enum):
            return Literal[tuple(col.type.enums)]  # noqa
        return cls.SQLA_TYPE_MAPPING.get(type(col.type), str)

    @classmethod
    def _parse_fields(
            cls, *,
            is_optional: Callable[[db.Column], bool] = None,
            is_excluded: Callable[[db.Column], bool] = None,
    ):
        fields = {}
        for col in cls.__table__.columns:  # noqa
            if is_excluded and is_excluded(col): continue
            optional = is_optional(col) if is_optional else col.nullable
            field_type = cls.get_field_type(col)
            fields[col.name] = (
                Optional[field_type] if optional
                else field_type, None if optional else ...
            )
        return fields

    @classmethod
    def _generate_pydantic_models(cls, model_name: str = None):
        if cls.__model__ and cls.__post_model__ and cls.__patch_model__: return
        model_name = model_name or cls.__name__.replace("Schema", "Model")

        relationship_fields = {}
        for relation in cls.__mapper__.relationships:  # noqa
            related_model_name = relation.mapper.class_.__name__.replace("Schema", "Model")
            if relation.mapper.class_.__model__:
                relationship_fields[relation.key] = (
                    Optional[list[relation.mapper.class_.__model__]] if relation.uselist
                    else Optional[relation.mapper.class_.__model__],
                    [] if relation.uselist else None
                )
            else:
                forward_ref = ForwardRef(related_model_name)
                relationship_fields[relation.key] = (
                    Optional[list[forward_ref]] if relation.uselist
                    else Optional[forward_ref],
                    [] if relation.uselist else None
                )

        fields = cls._parse_fields(is_optional=lambda col: col.nullable)
        cls.__model__ = create_model(model_name, **fields, **relationship_fields)

        fields = cls._parse_fields(is_optional=lambda col: col.nullable,
                                   is_excluded=lambda col: "post:exclude" in col.info.get("flags", {}))
        cls.__post_model__ = create_model(f"Post{model_name}", **fields)

        fields = cls._parse_fields(is_optional=lambda col: True,
                                   is_excluded=lambda col: not col.info.get("readonly", False) and
                                                           not col.primary_key and
                                                           "patch:exclude" in col.info.get("flags", {}))
        cls.__patch_model__ = create_model(f"Patch{model_name}", **fields)

    @classmethod
    def __declare_last__(cls):
        cls._generate_pydantic_models()
        event.listen(cls, "before_update", cls._readonly_update_listener)
        event.listen(cls, "before_insert", cls._uid_prepend_listener)

    # todo: fix readonly update listener
    @staticmethod
    def _readonly_update_listener(_, __, target):
        state = db.inspect(target)
        for attr in state.attrs:
            if isinstance(attr, db.orm.ColumnProperty):
                col = attr.columns[0]
                if getattr(col, "readonly", False):
                    history = state.get_history(col.key, passive=True)
                    if history.has_changes():
                        raise ValueError(f"Column '{col.key}' is readonly and cannot be modified.")

    @staticmethod
    def _uid_prepend_listener(_, __, target):
        if not target.uid:
            target.uid = f"{target.__tablename__.lower()}_{uuid.uuid4()}"

    def safe_getattr(self, attr: str):
        try:
            return getattr(self, attr)
        except MissingGreenlet:
            return None
        except DetachedInstanceError:
            return None

    @classmethod
    def model_load(cls, data: dict | BaseModel) -> "BaseSchema":
        data = data.model_dump() if isinstance(data, BaseModel) else data
        obj = cls()
        for col in cls.__table__.columns:  # noqa
            if col.key in data:
                setattr(obj, col.key, data[col.key])
        for relation in cls.__mapper__.relationships:  # noqa
            if relation.key in data:
                rel_schema = relation.mapper.class_
                rel_data = data[relation.key]
                if rel_data is None:
                    setattr(obj, relation.key, None)
                elif relation.uselist:
                    setattr(obj, relation.key, [rel_schema.model_load(rel) for rel in rel_data])
                else:
                    setattr(obj, relation.key, rel_schema.model_load(rel_data))
        return obj

    # todo: use exclude_none, exclude_defaults, exclude_unset
    def model_dump(self, *exclude, sanitize=False, use_model=False) -> dict | BaseModel:
        dump = {}
        for col in self.__table__.columns:  # noqa
            if col.key in exclude: continue
            dump[col.key] = self.safe_getattr(col.key)
        for relation in self.__mapper__.relationships:  # noqa
            if relation.key in exclude: continue
            rel = self.safe_getattr(relation.key)
            if rel is None:
                dump[relation.key] = None
            elif relation.uselist:
                dump[relation.key] = ([
                    related.model_dump(*exclude, relation.back_populates or relation.backref,
                                       sanitize=sanitize, use_model=use_model) for related in rel
                ])
            else:
                dump[relation.key] = rel.model_dump(
                    *exclude, relation.back_populates or relation.backref, sanitize=sanitize, use_model=use_model
                )
        dump = {k: v for k, v in dump.items() if not sanitize or (v not in (None, []))}
        if use_model: return self.__model__(**dump)
        return dump

    @classmethod
    def _hash(cls, value: bytes, *, salt: bytes = None) -> bytes:
        salt = salt or secrets.token_bytes(cls.SALT_SIZE)
        value = hashlib.pbkdf2_hmac("sha256", value, salt, 100000)
        return salt + value

    @classmethod
    def _hash_str(cls, value: str, *, salt: str = None) -> str:
        if salt: salt = bytes.fromhex(salt)
        hashed = cls._hash(value.encode(), salt=salt).hex()
        salt, hashed = cls._split_hash(bytes.fromhex(hashed))
        return f"{salt.hex()}:{hashed.hex()}"

    @classmethod
    def _split_hash(cls, hashed: bytes) -> tuple[bytes, bytes]:
        return hashed[:cls.SALT_SIZE], hashed[cls.SALT_SIZE:]

    @classmethod
    def _split_hash_str(cls, hashed: str) -> tuple[str, str]:
        hashed = hashed.split(":")
        return hashed[0], hashed[1]


class ListModel[ModelType: BaseSchema](BaseModel):
    items: list[ModelType]
    count: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(
        self,
        *,
        mode: Literal['json', 'python'] | str = 'python',
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        dump = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        dump["items"] = [item.model_dump() for item in self.items]
        return dump


class GenericManager[SchemaType: BaseSchema]:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.session_factory = sessionmaker(  # noqa
            bind=self.engine, class_=AsyncSession, expire_on_commit=False,
        )
        self.entity_manager = EntityManager(engine) if not isinstance(self, EntityManager) else None
        self.Schema._generate_pydantic_models()  # noqa

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(BaseSchema.metadata.create_all)

    @property
    def Schema(self) -> type[SchemaType]:
        return get_args(self.__orig_bases__[0])[0]  # noqa

    async def _flush(self, session: AsyncSession):
        try:
            await session.flush()
        except IntegrityError as e:
            raise GenericSchemaException(409, DBErrorCode.DB_INTEGRITY_ERROR, self.Schema, extras=e)

    def _resolve_joins(self, joins: NESTED_JOINS, *, loader=None):
        if isinstance(joins, QueryableAttribute):
            yield joinedload(joins) if loader is None else loader.subqueryload(joins)
            return
        parent = joins[0]
        children = joins[1:]
        loader = joinedload(parent) if loader is None else loader.subqueryload(parent)
        for child in children: yield from self._resolve_joins(child, loader=loader)

    def _ops(
            self,
            query: db.Select = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> db.Select[tuple[SchemaType]]:
        assert not (include and exclude), "parameters `include` and `exclude` are mutually exclusive"
        joins = joins or []
        query = db.select(self.Schema) if query is None else query
        joined = []
        for join in joins: joined.extend(self._resolve_joins(join))
        if joins: query = query.options(*joined)
        if include: query = query.options(load_only(
            *(getattr(self.Schema, col.name) for col in self.Schema.__table__.columns if col.name in include))  # noqa
        )
        if exclude: query = query.options(defer(*(getattr(self.Schema, col) for col in exclude)))
        return query

    @classmethod
    async def _filter(cls, query: db.Select, filters: NESTED_FILTERS, schema: type[BaseSchema]) -> db.Select:
        operator_mapping = {
            '==': lambda col, val: col == val,
            '!=': lambda col, val: col != val,
            '>': lambda col, val: col > val,
            '>=': lambda col, val: col >= val,
            '<': lambda col, val: col < val,
            '<=': lambda col, val: col <= val,
            'in': lambda col, val: col.in_(val if isinstance(val, list) else [val]),
            'between': lambda col, val: col.between(val[0], val[1])
        }

        relationship_filters = {}
        for relation in schema.__mapper__.relationships:  # noqa
            if relation.key in filters:
                relationship_filters[relation.key] = filters.pop(relation.key)

        for column, condition in filters.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op in operator_mapping:
                        query = query.filter(operator_mapping[op](getattr(schema, column), value))
            if isinstance(condition, list):
                query = query.filter(getattr(schema, column).in_(condition))
            else:
                query = query.filter(getattr(schema, column) == condition)  # noqa

        for relation_key, related_filter in relationship_filters.items():
            relation = schema.__mapper__.relationships[relation_key]
            related_model = relation.mapper.class_
            related_alias = aliased(related_model)
            query = query.join(related_alias)

            if not relation.uselist:
                query = await cls._filter(query, related_filter, related_model)
            else:
                if isinstance(related_filter, dict):
                    # if related_filter is a dict then all the related models must match the filter
                    subquery = db.select(related_model)
                    subquery = await cls._filter(subquery, related_filter, related_model)
                    condition = subquery.whereclause
                    if condition is not None:
                        # This condition means: there is no related record that does not match.
                        query = query.filter(~getattr(schema, relation_key).any(~condition))
                elif isinstance(related_filter, list):
                    # if related_filter is a list then any of the related models must match each filter
                    conditions = []
                    for sub_filter in related_filter:
                        subquery = await cls._filter(db.select(related_model), sub_filter, related_model)
                        sub_condition = subquery.whereclause
                        if sub_condition is not None:
                            # .any(sub_condition) will be True if there exists a related record matching sub_condition.
                            conditions.append(getattr(schema, relation_key).any(sub_condition))
                    if conditions:
                        # All filter dictionaries must have a matching related record.
                        query = query.filter(db.and_(*conditions))

        return query

    async def create(
            self,
            data: SchemaType | BaseModel,
            *,
            upstreamId: str = None,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            record = self.Schema.model_load(data) if isinstance(data, BaseModel) else data
            session.add(record)
            await self._flush(session)
            if upstreamId: await self.entity_manager.create(
                self.entity_manager.Schema(
                    upstreamId=upstreamId, tablename=self.Schema.__tablename__, downstreamId=record.uid
                ),
                session=session,
            )
            return await self.fetch(
                record.uid,  # noqa
                session=session,
                joins=joins,
                include=include,
                exclude=exclude
            )
        async with self.session_factory() as session:
            record = await GenericManager.create(
                self,
                data,
                upstreamId=upstreamId,
                session=session,
                joins=joins,
                include=include,
            )
            await session.commit()
            return record

    async def create_all(
            self,
            data: list[SchemaType | BaseModel],
            *,
            upstreamId: str = None,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> ListModel[SchemaType]:
        if session:
            data = [self.Schema.model_load(d) if isinstance(d, BaseModel) else d for d in data]
            session.add_all(data)
            await self._flush(session)
            if upstreamId: await asyncio.gather(*(self.entity_manager.create(self.entity_manager.Schema(
                upstreamId=upstreamId, tablename=self.Schema.__tablename__, downstreamId=d.uid
            ), session=session) for d in data))
            return await self.fetch_all(
                filters={"uid": [d.uid for d in data]},  # noqa
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
        async with self.session_factory() as session:
            records = await GenericManager.create_all(
                self,
                data,
                upstreamId=upstreamId,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            await session.commit()
            return records

    async def fetch(
            self,
            uid: str,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            query = db.select(self.Schema).filter_by(uid=uid)
            query = self._ops(query, joins, include, exclude)
            record = await session.execute(query)
            record = record.unique().scalar_one_or_none()
            if record is None: raise GenericSchemaException(404, DBErrorCode.DB_NOT_FOUND, self.Schema)
            return record
        async with self.session_factory() as session:
            return await GenericManager.fetch(
                self,
                uid,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )

    async def fetch_all(
            self,
            limit: int = 0,
            offset: int = 0,
            filters: NESTED_FILTERS = None,
            sorts: list[str] = None,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> ListModel[SchemaType]:
        if session:
            await self._flush(session)
            query = db.select(self.Schema)
            if limit: query = query.limit(limit)
            query = query.offset(offset)
            query = self._ops(query, joins, include, exclude)
            if filters: query = await self._filter(query, filters, self.Schema)
            if sorts:
                for sort in sorts:
                    acs = sort.startswith("-")
                    sort = sort.lstrip("-").lstrip("+")
                    col = getattr(self.Schema, sort)
                    query = query.order_by(col.asc() if acs else col.desc())
            records = await session.execute(query)
            records = list(records.unique().scalars())
            return ListModel[self.Schema](items=records, count=len(records))
        async with self.session_factory() as session:
            return await GenericManager.fetch_all(
                self,
                limit,
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )

    async def fetch_one(
            self,
            offset: int = 0,
            filters: NESTED_FILTERS = None,
            sorts: list[str] = None,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        records = await self.fetch_all(
            1,
            offset,
            filters,
            sorts,
            session=session,
            joins=joins,
            include=include,
            exclude=exclude,
        )
        if not records: raise GenericSchemaException(404, DBErrorCode.DB_NOT_FOUND, self.Schema)
        return records.items[0]

    async def create_or_fetch(
            self,
            data: SchemaType | BaseModel,
            *,
            upstreamId: str = None,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
            ignore: list[str] = None,
    ) -> SchemaType:
        if session:
            data = self.Schema.model_load(data) if isinstance(data, BaseModel) else data
            for col in ignore or []: setattr(data, col, None)
            data_dump = data.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True) \
                if isinstance(data, BaseModel) \
                else data.model_dump(sanitize=True)

            query = db.select(self.Schema)
            query = await self._filter(query, data_dump, self.Schema)

            record = await session.execute(query)
            try:
                record = record.unique().scalar_one_or_none()
            except MultipleResultsFound:
                raise GenericSchemaException(status_code=409, error_enum=DBErrorCode.DB_MULTIPLE_FOUND,
                                             schema=self.Schema)

            if record is None:
                record = await self.create(
                    data, upstreamId=upstreamId, session=session, joins=joins, include=include, exclude=exclude
                )

            return record

        async with self.session_factory() as session:
            record = await GenericManager.create_or_fetch(
                self,
                data,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
                ignore=ignore
            )
            await session.commit()
            return record

    async def update(
            self,
            uid: str,
            updates: dict,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            query = db.select(self.Schema).filter_by(uid=uid)
            query = self._ops(query, joins, include, exclude)
            record = await session.execute(query)
            record = record.unique().scalar_one_or_none()
            if record is None: raise GenericSchemaException(404, DBErrorCode.DB_NOT_FOUND, self.Schema)
            for col, val in updates.items(): setattr(record, col, val)
            await self._flush(session)
            return record
        async with self.session_factory() as session:
            record = await GenericManager.update(
                self,
                uid,
                updates,
                joins=joins,
                session=session,
                include=include,
                exclude=exclude,
            )
            await session.commit()
            return record

    async def update_all(
            self,
            limit: int = 1,
            offset: int = 0,
            filters: NESTED_FILTERS = None,
            sorts: list[str] = None,
            *,
            updates: dict,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> ListModel[SchemaType]:
        if session:
            records = await self.fetch_all(
                limit,
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude
            )
            if not records: raise GenericSchemaException(404, DBErrorCode.DB_NOT_FOUND, self.Schema)
            # noinspection PyUnresolvedReferences
            await session.execute(
                db.update(self.Schema).where(self.Schema.uid.in_([record.uid for record in records.items])).values(
                    **updates)
            )
            return records
        async with self.session_factory() as session:
            records = await GenericManager.update_all(
                self,
                limit,
                offset,
                filters,
                sorts,
                updates=updates,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            await session.commit()
            return records

    async def update_one(
            self,
            offset: int = 0,
            filters: NESTED_FILTERS = None,
            sorts: list[str] = None,
            *,
            updates: dict,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            records = await self.update_all(
                1,
                offset,
                filters,
                sorts,
                updates=updates,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            if not records: raise GenericSchemaException(404, DBErrorCode.DB_NOT_FOUND, self.Schema)
            return records.items[0]
        async with self.session_factory() as session:
            return await GenericManager.update_one(
                self,
                offset,
                filters,
                sorts,
                updates=updates,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )

    async def delete(
            self,
            uid: str,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            return await self.delete_one(
                filters={"uid": uid},
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
        async with self.session_factory() as session:
            record = await GenericManager.delete(
                self,
                uid,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            await session.commit()
            return record

    async def delete_all(
            self,
            limit: int = 1,
            offset: int = 0,
            filters: NESTED_FILTERS = None,
            sorts: list[str] = None,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> ListModel[SchemaType]:
        if session:
            records = await self.fetch_all(
                limit,
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            try:
                if self.entity_manager: await self.entity_manager.delete_all(
                    filters={"tablename": self.Schema.__tablename__,
                             "downstreamId": [record.uid for record in records.items]},
                    session=session
                )
            except GenericSchemaException:
                pass
            await session.execute(
                db.delete(self.Schema).where(self.Schema.uid.in_([record.uid for record in records.items]))  # noqa
            )
            await self._flush(session)
            return records
        async with self.session_factory() as session:
            records = await GenericManager.delete_all(
                self,
                limit,
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            await session.commit()
            return records

    async def delete_one(
            self,
            offset: int = 0,
            filters: dict[str, list[str] | str] = None,
            sorts: list[str] = None,
            *,
            session: AsyncSession = None,
            joins: list[NESTED_JOINS] = None,
            include: list[str] = None,
            exclude: list[str] = None,
    ) -> SchemaType:
        if session:
            record = await self.fetch_one(
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
                include=include,
                exclude=exclude,
            )
            try:
                if self.entity_manager: await self.entity_manager.delete_one(
                    filters={"tablename": self.Schema.__tablename__, "downstreamId": record.uid}, session=session
                )
            except GenericSchemaException:
                pass
            await session.delete(record)
            await self._flush(session)
            return record
        async with self.session_factory() as session:
            record = await GenericManager.delete_one(
                self,
                offset,
                filters,
                sorts,
                session=session,
                joins=joins,
            )
            await session.commit()
            return record


class EntitySchema(BaseSchema):
    __tablename__ = 'entities'
    __table_args__ = (db.UniqueConstraint('tablename', 'downstreamId'),)

    upstreamId = db.Column(db.String, nullable=False)
    tablename = db.Column(db.String, nullable=False)
    downstreamId = db.Column(db.String, nullable=False)


class EntityManager(GenericManager[EntitySchema]):
    async def fetch_links(self, upstreamId: str) -> dict[str, list[str]]:
        entities = await self.fetch_all(filters={"upstreamId": upstreamId})
        links = defaultdict(list)
        for entity in entities.items: links[entity.tablename].append(entity.downstreamId)
        return links


__all__ = [
    "BaseSchema", "GenericManager",
    "EntitySchema", "EntityManager",
]
