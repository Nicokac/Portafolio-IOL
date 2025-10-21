from shared import version


def test_version_metadata_exposes_expected_fields():
    info = version.get_version_info()

    assert info["version"] == version.__version__
    assert info["codename"] == version.__codename__
    assert info["release_date"] == version.__release_date__
    assert info["build_signature"] == version.__build_signature__
    assert info["stability"] == version.__stability__
    assert info["changelog_ref"] == version.__changelog_ref__
    assert version.DEFAULT_VERSION == version.__version__
    assert isinstance(version.BUILD_SIGNATURE, str)
    assert version.BUILD_SIGNATURE
