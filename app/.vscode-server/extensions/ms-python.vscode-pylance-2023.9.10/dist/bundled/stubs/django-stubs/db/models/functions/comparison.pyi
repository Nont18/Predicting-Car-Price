from typing import Any

from django.db.models import Func
from django.db.models.fields import Field

class Cast(Func):
    def __init__(
        self, expression: Any, output_field: str | Field[Any, Any]
    ) -> None: ...

class Coalesce(Func): ...
class Greatest(Func): ...
class Least(Func): ...
class NullIf(Func): ...