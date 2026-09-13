"""Finding the three mounts of one sample from one file name."""

from clayquant.pairing import find_siblings, mount_of, sample_key


TRIPLET = [
    "DBB_17_air.xrdml",
    "DBB_17_eg.xrdml",
    "DBB_17_heat.xrdml",
    "DBB_17_PW_air.xrdml",
    "DBB_17_PW_eg.xrdml",
    "DBB_17_PW_heat.xrdml",
    "DBB_17_.txt",
    "Clinochlore.cif",
]


def test_one_file_identifies_the_other_two():
    assert find_siblings("DBB_17_air.xrdml", TRIPLET) == {
        "air": "DBB_17_air.xrdml",
        "glycol": "DBB_17_eg.xrdml",
        "heated": "DBB_17_heat.xrdml",
    }


def test_any_of_the_three_is_a_valid_starting_point():
    from_glycol = find_siblings("DBB_17_PW_eg.xrdml", TRIPLET)
    from_heated = find_siblings("DBB_17_PW_heat.xrdml", TRIPLET)
    assert from_glycol == from_heated
    assert from_glycol["air"] == "DBB_17_PW_air.xrdml"


def test_a_longer_name_is_not_matched_by_a_shorter_one():
    """DBB_17 and DBB_17_PW are different samples, not two spellings of one."""
    assert find_siblings("DBB_17_air.xrdml", TRIPLET)["glycol"] == "DBB_17_eg.xrdml"
    assert sample_key("DBB_17_air.xrdml") != sample_key("DBB_17_PW_air.xrdml")


def test_the_extension_of_the_chosen_file_is_kept():
    both = ["S_air.xrdml", "S_air.xy", "S_eg.xrdml", "S_eg.xy", "S_heat.xrdml", "S_heat.xy"]
    assert find_siblings("S_air.xy", both) == {
        "air": "S_air.xy", "glycol": "S_eg.xy", "heated": "S_heat.xy"
    }
    assert find_siblings("S_air.xrdml", both)["glycol"] == "S_eg.xrdml"


def test_a_mount_present_only_in_another_format_is_still_found():
    mixed = ["S_air.xrdml", "S_eg.xy", "S_heat.xrdml"]
    assert find_siblings("S_air.xrdml", mixed)["glycol"] == "S_eg.xy"


def test_a_missing_mount_is_simply_absent():
    assert set(find_siblings("S_air.xy", ["S_air.xy", "S_eg.xy"])) == {"air", "glycol"}


def test_a_name_without_a_mount_token_matches_nothing():
    assert find_siblings("Clinochlore.cif", TRIPLET) == {}
    assert mount_of("DBB_17_.txt") is None


def test_the_token_must_stand_alone():
    """A substring test would read the 'ad' of 'Bad' as an air-dried mount."""
    assert mount_of("Bad_Segeberg_eg.xy") == "glycol"
    assert sample_key("Bad_Segeberg_eg.xy") == "bad_segeberg|"
    assert mount_of("Badenweiler.xy") is None


def test_the_last_token_wins():
    """'EG' names the mount; an earlier coincidence does not."""
    assert mount_of("air_quality_site_eg.xy") == "glycol"


def test_alternative_spellings():
    assert mount_of("S_AD.xy") == "air"
    assert mount_of("S_glycol.xy") == "glycol"
    assert mount_of("S_550.xy") == "heated"
    assert mount_of("S_lutro.xy") == "air"


def test_a_separator_other_than_underscore():
    hyphenated = ["S-01-air.xy", "S-01-eg.xy", "S-01-heat.xy"]
    assert find_siblings("S-01-eg.xy", hyphenated)["heated"] == "S-01-heat.xy"


def test_a_suffix_after_the_mount_token_is_part_of_the_key():
    scans = ["S_air_1.xy", "S_eg_1.xy", "S_eg_2.xy", "S_heat_1.xy"]
    found = find_siblings("S_air_1.xy", scans)
    assert found["glycol"] == "S_eg_1.xy"
    assert found["heated"] == "S_heat_1.xy"
