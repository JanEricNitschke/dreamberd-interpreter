import dataclasses
from collections.abc import Callable
from typing import Any, assert_never

from dreamberd.base import NonFormattedError, Token, TokenType

# Need to import everything for the eval calls.
from dreamberd.builtin import (
    KEYWORDS,
    BuiltinFunction,
    DreamberdBoolean,
    DreamberdFunction,
    DreamberdKeyword,
    DreamberdList,
    DreamberdMap,
    DreamberdNumber,
    DreamberdObject,
    DreamberdPromise,
    DreamberdSpecialBlankValue,
    DreamberdString,
    DreamberdUndefined,
    DreamberdValue,
    Name,
    Variable,
    db_list_pop,
    db_list_push,
    db_str_pop,
    db_str_push,
)
from dreamberd.processor.syntax_tree import (
    AfterStatement,
    ClassDeclaration,
    CodeStatement,
    Conditional,
    DeleteStatement,
    ExportStatement,
    ExpressionStatement,
    FunctionDefinition,
    ImportStatement,
    ReturnStatement,
    ReverseStatement,
    VariableAssignment,
    VariableDeclaration,
    WhenStatement,
)

SerializedDict = dict[str, str | dict | list]
DataclassSerializations = Name | Variable | DreamberdValue | CodeStatement | Token


def serialize_obj(obj: Any) -> SerializedDict:
    match obj:
        case Name() | Variable() | DreamberdValue() | CodeStatement() | Token():
            return serialize_dreamberd_obj(obj)
        case _:
            return serialize_python_obj(obj)


def deserialize_obj(val: dict) -> Any:
    if "dreamberd_obj_type" in val:
        return deserialize_dreamberd_obj(val)
    if "python_obj_type" in val:
        return deserialize_python_obj(val)
    msg = "Invalid object type in Dreamberd Variable deserialization."
    raise NonFormattedError(msg)


def serialize_python_obj(obj: Any) -> dict[str, str | dict | list]:
    match obj:
        case TokenType():
            val = obj.value
        case dict():
            if not all(isinstance(k, str) for k in obj):
                msg = "Serialization Error: Encountered non-string dictionary keys."
                raise NonFormattedError(msg)
            val = {k: serialize_obj(v) for k, v in obj.items()}
        case list() | tuple():
            val = [serialize_obj(x) for x in obj]
        case str():
            val = obj
        case None | int() | float() | bool():
            val = str(obj)
        case func if isinstance(func, Callable):
            val = func.__name__
        case _:
            assert_never(obj)
    return {
        "python_obj_type": type(obj).__name__
        if not isinstance(obj, Callable)
        else "function",
        "value": val,
    }


def deserialize_python_obj(val: dict) -> Any:
    if val["python_obj_type"] not in [
        "int",
        "float",
        "dict",
        "function",
        "list",
        "tuple",
        "str",
        "TokenType",
        "NoneType",
        "bool",
    ]:
        print(val["python_obj_type"])
        msg = "Invalid `python_obj_type` detected in deserialization."
        raise NonFormattedError(msg)

    match val["python_obj_type"]:
        case "list":
            return [deserialize_obj(x) for x in val["value"]]
        case "tuple":
            return tuple(deserialize_obj(x) for x in val["value"])
        case "dict":
            return {k: deserialize_obj(v) for k, v in val["value"].items()}
        case "int" | "float" | "str":
            return eval(val["python_obj_type"])(val["value"])  # RAISES ValueError
        case "NoneType":
            return None
        case "bool":
            if val["value"] not in ["True", "False"]:
                msg = "Invalid boolean detected in object deserialization."
                raise NonFormattedError(msg)
            return eval(val["value"])
        case "TokenType":
            if v := TokenType.from_val(val["value"]):
                return v
            msg = "Invalid TokenType detected in object deserialization."
            raise NonFormattedError(msg)
        case "function":
            if val["value"] in [
                "db_list_pop",
                "db_list_push",
                "db_str_pop",
                "db_str_push",
            ]:
                return eval(val["value"])  # trust me bro this is W code
            if not (v := KEYWORDS.get(val["value"])) or not isinstance(
                v.value, BuiltinFunction
            ):
                msg = "Invalid builtin function detected in object deserialization."
                raise NonFormattedError(msg)
            return v.value.function
        case invalid:
            assert_never(invalid)


def serialize_dreamberd_obj(
    val: DataclassSerializations,
) -> dict[str, str | dict | list]:
    return {
        "dreamberd_obj_type": type(val).__name__,
        "attributes": [
            {"name": field.name, "value": serialize_obj(getattr(val, field.name))}
            for field in dataclasses.fields(val)
        ],
    }


def get_subclass_name_list(cls: type[DataclassSerializations]) -> list[str]:
    return [*(x.__name__ for x in cls.__subclasses__())]


def deserialize_dreamberd_obj(val: dict) -> DataclassSerializations:
    if val["dreamberd_obj_type"] not in [
        "Name",
        "Variable",
        "Token",
        *get_subclass_name_list(CodeStatement),
        *get_subclass_name_list(DreamberdValue),
    ]:
        msg = "Invalid `dreamberd_obj_type` detected in deserialization."
        raise NonFormattedError(msg)

    # beautiful, elegant, error-free, safe python code :D
    attrs = {at["name"]: deserialize_obj(at["value"]) for at in val["attributes"]}
    return eval(val["dreamberd_obj_type"])(**attrs)


if __name__ == "__main__":
    list_test_case = DreamberdList(
        [
            DreamberdString("Hello world!"),
            DreamberdNumber(123.45),
        ]
    )
    serialized = serialize_obj(list_test_case)
    __import__("pprint").pprint(serialized)
    assert list_test_case == deserialize_obj(serialized)  # noqa: S101
