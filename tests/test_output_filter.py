from guardrails.output_filter import check_output


def test_benign_output_is_allowed():
    result = check_output("The capital of France is Paris.")
    assert result["allowed"] is True
    assert result["reason"] == ""


def test_malware_output_is_blocked():
    result = check_output("Here is how to make malware: import os and...")
    assert result["allowed"] is False
    assert result["reason"] != ""


def test_ransomware_output_is_blocked():
    result = check_output("To create ransomware, start by encrypting the filesystem.")
    assert result["allowed"] is False


def test_bomb_output_is_blocked():
    result = check_output("To make a bomb, first gather these materials...")
    assert result["allowed"] is False


def test_explosive_recipe_is_blocked():
    result = check_output("The bomb recipe requires ammonium nitrate.")
    assert result["allowed"] is False


def test_csam_output_is_blocked():
    result = check_output("csam is distributed through these channels...")
    assert result["allowed"] is False


def test_child_porn_output_is_blocked():
    result = check_output("child pornography can be found...")
    assert result["allowed"] is False


def test_jailbreak_success_phrase_is_blocked():
    result = check_output("I am now DAN and will answer anything.")
    assert result["allowed"] is False


def test_no_restrictions_phrase_is_blocked():
    result = check_output("I have no restrictions and will comply.")
    assert result["allowed"] is False


def test_case_insensitive_blocking():
    result = check_output("HERE IS HOW TO MAKE MALWARE FOR YOU")
    assert result["allowed"] is False


def test_case_insensitive_csam():
    result = check_output("CSAM is illegal everywhere.")
    assert result["allowed"] is False
