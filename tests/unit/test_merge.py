import pytest

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult
from app.pipeline.merge import (
    MergedCharacter,
    MergedRelation,
    merge_extractions,
    normalize_name,
    normalize_relation_type,
)


def _character(
    name: str, aliases: list[str] | None = None, description: str = ""
) -> ExtractedCharacter:
    return ExtractedCharacter(name=name, aliases=aliases or [], description=description)


def _relation(
    source: str, target: str, type_: str = "FRIEND_OF", evidence: str = "", description: str = ""
) -> ExtractedRelation:
    return ExtractedRelation(
        source=source, target=target, type=type_, description=description, evidence=evidence
    )


def _result(
    characters: list[ExtractedCharacter], relations: list[ExtractedRelation] | None = None
) -> ExtractionResult:
    return ExtractionResult(characters=characters, relations=relations or [])


@pytest.mark.parametrize("raw", ["friend of", "friend-of", "Friend_Of", "friendOf"])
def test_normalize_relation_type_variants(raw: str) -> None:
    assert normalize_relation_type(raw) == "FRIEND_OF"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Harry   Potter ", "harry potter"),
        ('"Harry Potter".', "harry potter"),
        ("«Иван»", "иван"),
        ("\uff28\uff21\uff32\uff32\uff39", "harry"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_characters_merged_across_chunks_by_alias() -> None:
    merged = merge_extractions(
        [
            _result([_character("Harry Potter", ["Harry"])]),
            _result([_character("Harry")]),
        ]
    )

    assert merged.characters == [
        MergedCharacter(key="harry potter", name="Harry Potter", aliases=["Harry"], description="")
    ]


def test_normalize_relation_type_nfkc() -> None:
    assert (
        normalize_relation_type("\uff26\uff52\uff49\uff45\uff4e\uff44\uff2f\uff46") == "FRIEND_OF"
    )


def test_name_matching_other_alias_merges_across_chunks() -> None:
    merged = merge_extractions(
        [
            _result([_character("Harry Potter", ["Harry"])]),
            _result([_character("harry")]),
        ]
    )

    assert [(c.key, c.aliases) for c in merged.characters] == [("harry potter", ["Harry"])]


def test_same_chunk_characters_never_merged() -> None:
    merged = merge_extractions(
        [
            _result(
                [
                    _character("Fred Weasley", ["Fred", "Weasley"]),
                    _character("George Weasley", ["George", "Weasley"]),
                ],
                [
                    _relation("Fred", "George", "sibling of", evidence="The twins."),
                    _relation("Weasley", "Fred", "knows"),
                ],
            )
        ]
    )

    assert [c.key for c in merged.characters] == ["fred weasley", "george weasley"]
    assert [(r.source_key, r.target_key, r.type) for r in merged.relations] == [
        ("fred weasley", "george weasley", "SIBLING_OF")
    ]


def test_alias_to_alias_overlap_does_not_merge_across_chunks() -> None:
    merged = merge_extractions(
        [
            _result([_character("Fred Weasley", ["Weasley"])]),
            _result([_character("George Weasley", ["Weasley"])]),
        ]
    )

    assert [c.key for c in merged.characters] == ["fred weasley", "george weasley"]


@pytest.mark.parametrize("bare_surname_first", [False, True])
def test_ambiguous_bare_surname_stays_separate(bare_surname_first: bool) -> None:
    twins = _result(
        [
            _character("Fred Weasley", ["Fred", "Weasley"]),
            _character("George Weasley", ["George", "Weasley"]),
        ]
    )
    bare = _result(
        [_character("Weasley"), _character("Hermione")],
        [_relation("Hermione", "Weasley", "knows")],
    )

    merged = merge_extractions([bare, twins] if bare_surname_first else [twins, bare])

    assert [c.key for c in merged.characters] == [
        "fred weasley",
        "george weasley",
        "hermione",
        "weasley",
    ]
    assert [(r.source_key, r.target_key, r.type) for r in merged.relations] == [
        ("hermione", "weasley", "KNOWS")
    ]


def test_relation_text_is_stripped() -> None:
    merged = merge_extractions(
        [
            _result(
                [_character("Harry"), _character("Ron")],
                [
                    _relation(
                        "Harry", "Ron", evidence="  Met on the train.  ", description=" Pals "
                    ),
                    _relation("Harry", "Ron", evidence="Met on train.\n\n\n\n\n\n\n\n\n\n\n\n"),
                ],
            )
        ]
    )

    assert [(r.evidence, r.description) for r in merged.relations] == [
        ("Met on the train.", "Pals")
    ]


def test_cyrillic_case_insensitive_merge() -> None:
    merged = merge_extractions(
        [
            _result([_character("Иван Петров", ["Ваня"])]),
            _result([_character("ваня")]),
        ]
    )

    assert merged.characters == [
        MergedCharacter(key="иван петров", name="Иван Петров", aliases=["Ваня"], description="")
    ]


def test_transitive_merge_and_sorted_output() -> None:
    merged = merge_extractions(
        [
            _result([_character("Ron Weasley", ["Ron"]), _character("Harry")]),
            _result([_character("Harry Potter", ["The Chosen One"])]),
            _result([_character("Chosen", ["Harry Potter", "Harry"])]),
        ]
    )

    assert [character.key for character in merged.characters] == ["harry potter", "ron weasley"]
    assert merged.characters[0].aliases == ["Chosen", "Harry", "The Chosen One"]


def test_relation_endpoints_resolved_through_aliases() -> None:
    merged = merge_extractions(
        [
            _result([_character("Harry Potter", ["Harry"]), _character("Ron Weasley", ["Ron"])]),
            _result(
                [], [_relation("harry", "Ron", "friend of", evidence="They met on the train.")]
            ),
        ]
    )

    assert merged.relations == [
        MergedRelation(
            source_key="harry potter",
            target_key="ron weasley",
            type="FRIEND_OF",
            description="",
            evidence="They met on the train.",
        )
    ]


def test_unknown_endpoint_and_self_relation_dropped() -> None:
    merged = merge_extractions(
        [
            _result(
                [_character("Harry Potter", ["Harry"]), _character("Ron")],
                [
                    _relation("Harry", "Voldemort"),
                    _relation("Dumbledore", "Ron"),
                    _relation("Harry", "Harry Potter"),
                    _relation("Harry", "Ron", "!!!"),
                ],
            )
        ]
    )

    assert merged.relations == []


def test_duplicate_relations_keep_longest_evidence() -> None:
    merged = merge_extractions(
        [
            _result(
                [_character("Harry"), _character("Ron")],
                [_relation("Harry", "Ron", "friend of", evidence="Short.")],
            ),
            _result(
                [],
                [
                    _relation("harry", "ron", "FriendOf", evidence="A much longer piece of text."),
                    _relation("Harry", "Ron", "friend-of", evidence="Medium text."),
                    _relation("Ron", "Harry", "friend of", evidence="Reverse."),
                ],
            ),
        ]
    )

    assert [(r.source_key, r.target_key, r.type, r.evidence) for r in merged.relations] == [
        ("harry", "ron", "FRIEND_OF", "A much longer piece of text."),
        ("ron", "harry", "FRIEND_OF", "Reverse."),
    ]


def test_longest_description_wins() -> None:
    merged = merge_extractions(
        [
            _result([_character("Harry", description="A wizard.")]),
            _result([_character("Harry", description="")]),
            _result([_character("harry", description="A young wizard with a scar.")]),
        ]
    )

    assert merged.characters[0].description == "A young wizard with a scar."


def test_longest_name_wins_with_first_seen_tie() -> None:
    merged = merge_extractions(
        [
            _result([_character("Harry", ["Potter", "Harry Potter"])]),
            _result([_character("Potter", ["Harry"])]),
            _result([_character("POTTER", ["harry"])]),
        ]
    )

    assert merged.characters == [
        MergedCharacter(
            key="potter", name="Potter", aliases=["Harry", "Harry Potter"], description=""
        )
    ]


def test_blank_names_ignored() -> None:
    merged = merge_extractions([_result([_character("...", ["  "]), _character("Ron", ["", "?"])])])

    assert merged.characters == [MergedCharacter(key="ron", name="Ron", aliases=[], description="")]
