from dataclasses import dataclass

@dataclass(frozen=True)
class Principal:
    subject: str = "demo-operator"
    tenant: str = "demo-workspace"
    groups: tuple[str, ...] = ("workspace",)
    admin: bool = True

def principal() -> Principal:
    return Principal()