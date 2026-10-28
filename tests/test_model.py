import pytest

from x3d_publisher import model as models


def test_bundled_ww3():
    assert "ww3_4km" in models.bundled()
    m = models.load("ww3_4km")
    assert m.name == "ww3_4km" and m.url.startswith("https://thredds.met.no/")
    assert m.map_shard == (1, 1280, 768) and m.ov_factor == 4
    assert m.scales["hs"] == 0.01 and set(m.vars) <= set(m.scales) and set(m.ov_vars) <= set(m.scales)


def test_bundled_models_hold_no_secrets():
    from importlib import resources

    for n in models.bundled():
        text = resources.files("x3d_publisher.models").joinpath(f"{n}.toml").read_text().lower()
        assert not any(w in text for w in ("secret", "password", "token", "access_key", "s3://"))


def test_every_variable_needs_a_scale():
    with pytest.raises(ValueError, match="no scale"):
        models.Model.from_dict({"name": "x", "url": "u", "vars": ["a"], "ov_vars": ["a"], "scales": {}})


def test_unknown_model():
    with pytest.raises(SystemExit):
        models.load("nope")


def test_model_from_file(tmp_path):
    f = tmp_path / "m.toml"
    f.write_text('name = "m"\nurl = "u"\nvars = ["a"]\nov_vars = ["a"]\ny_dim = "Y"\n[scales]\na = 0.5\n[layout]\nov_factor = 2\n')
    m = models.load(f)
    assert (m.name, m.y_dim, m.ov_factor, m.scales) == ("m", "Y", 2, {"a": 0.5})
