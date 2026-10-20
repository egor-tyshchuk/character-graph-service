from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from app.llm.schemas import DedupVerdict, ExtractionResult, ImageDescription


def _object_schemas(schema: dict[str, Any]) -> list[dict[str, Any]]:
    objects = [schema] if schema.get("type") == "object" else []
    for definition in schema.get("$defs", {}).values():
        objects.extend(_object_schemas(definition))
    return objects


@pytest.mark.parametrize("model", [ExtractionResult, ImageDescription, DedupVerdict])
def test_llm_output_schema_has_only_required_fields(model: type[BaseModel]) -> None:
    for schema in _object_schemas(to_strict_json_schema(model)):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert all("default" not in prop for prop in schema["properties"].values())
