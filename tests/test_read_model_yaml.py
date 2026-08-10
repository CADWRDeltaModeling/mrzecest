import pytest
from mrzecest.fitting_util import read_model_yaml


def test_read_model_yaml_fails_when_front_missing(tmp_path):
    # A model YAML lacking a 'front' block should raise
    yml = tmp_path / "model_no_front.yaml"
    content = """
version: 1
constants:
  sb_uS_cm: 200.0
  so_uS_cm: 55000.0
filter_setup:
  dt: 3h
  k0: 6
  filter_length: 1
  centering: causal
  afilt: [0.0]
g_model:
  beta_log10: 10.0
  npow: 0.5
physics:
  area_coef: 0.0
  energy_coef: 0.0
inner_linear:
  b0: 0.0
  b1: 0.0
"""
    yml.write_text(content)
    with pytest.raises(ValueError):
        read_model_yaml(str(yml))


def test_read_model_yaml_with_front_passes(tmp_path):
    # a small YAML with front should read OK
    yml = tmp_path / "model_with_front.yaml"
    content = """
version: 1
constants:
  sb_uS_cm: 200.0
  so_uS_cm: 55000.0
filter_setup:
  dt: 3h
  k0: 6
  filter_length: 1
  centering: causal
  afilt: [0.0]
g_model:
  beta_log10: 10.0
  npow: 0.5
  g_thr_tide: 20000.0
  width_frac_tide: 0.6
physics:
  area_coef: 0.0
  energy_coef: 0.0
inner_linear:
  b0: 0.0
  b1: 0.0
front:
  ec_target: 20000.0
  gthr: 1.0
  energy_ref: 1.0
  width_frac: 0.1
"""
    yml.write_text(content)
    model = read_model_yaml(str(yml))
    assert "front" in model
    for k in ("ec_target", "gthr", "energy_ref", "width_frac"):
        assert k in model["front"]
