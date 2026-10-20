import hashlib
import json

from app.llm.schemas import CharacterProfile
from app.pipeline.merge import normalize_name


def profile_hash(profile: CharacterProfile) -> str:
    identity = [
        normalize_name(profile.name),
        sorted(normalize_name(alias) for alias in profile.aliases),
        profile.description,
    ]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
