"""Read-only role views over the already verified September13 feature cache."""
from pilots.layer_screen_20260913.data import CachedDataset, materialize, atomic_torch
from .protocol import ARMS, ROLES, validate_inputs


class RoleDataset(CachedDataset):
    def __init__(self, cache, execution, roles, role, arm):
        if role not in ROLES or arm not in ARMS: raise ValueError("Unknown execution role or arm")
        self.execution,self.role_map=validate_inputs(cache,execution,roles)
        self.role=role
        spec=self.role_map["roles"][role]
        super().__init__(cache,spec["original_split"],arm)
        wanted=set(spec["record_ids"])
        self.entries=[row for row in self.entries if row["record_id"] in wanted]
        if [row["record_id"] for row in self.entries]!=spec["record_ids"]:
            raise RuntimeError("Role membership/order does not match the immutable index")
        if {row["image_id"] for row in self.entries}!=set(spec["photo_ids"]):
            raise RuntimeError("Role photo identity changed")
        # Parent metadata/split_id remains untouched. The role lives in config
        # and role_map, never in rewritten cache rows.
