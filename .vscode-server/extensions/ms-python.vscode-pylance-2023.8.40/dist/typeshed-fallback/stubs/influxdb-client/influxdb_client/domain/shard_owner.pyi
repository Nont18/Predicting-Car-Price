from _typeshed import Incomplete

class ShardOwner:
    openapi_types: Incomplete
    attribute_map: Incomplete
    discriminator: Incomplete
    def __init__(self, node_id: Incomplete | None = None) -> None: ...
    @property
    def node_id(self): ...
    @node_id.setter
    def node_id(self, node_id) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...