# Utilities for the library

import yaml
import pandas as pd
import os


def validate_fit_config(cfg: dict) -> None:
    """Validate fitting configuration.

    Parameters
    ----------
    cfg : dict
        Configuration data loaded from ``fitting_config.yaml``.

    Raises
    ------
    TypeError
        If ``cfg`` is not a mapping or if ``filter_setup`` is not a mapping.
    ValueError
        If required keys are missing or unsupported legacy keys are present.

    Notes
    -----
    Call this as soon as the YAML is parsed so incompatible layouts are caught
    before any fitting workflow runs.
    ```
    This project intentionally avoids backward-compat inference. If the config
    contains keys that are no longer part of the model spec, raise immediately.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"fit config must be a mapping, got {type(cfg)}")

    # storage_area was historically mentioned but the model uses `area_coef`.
    if "storage_area" in cfg:
        raise ValueError(
            "'storage_area' is not part of the model. Remove it from the config. "
            "The storage/area effect is represented by the optimized 'area_coef'."
        )

    # No legacy/backward constraints: fit configs must not include the old
    # 'param:' block or the old top-level fit_run.x0 / fit_run.bounds.
    if "param" in cfg:
        raise ValueError(
            "Legacy 'param:' block is not accepted. Specify outer optimization bounds/starts "
            "only via fit_run.outer_params."
        )

    fit_run = cfg.get("fit_run")
    if isinstance(fit_run, dict):
        if "x0" in fit_run or "bounds" in fit_run:
            raise ValueError(
                "Legacy fit_run.x0 / fit_run.bounds are not accepted. Use fit_run.outer_params instead."
            )

    for k in ("so", "sb", "filter_setup"):
        if k not in cfg:
            raise ValueError(f"fit config missing required key '{k}'")

    fs = cfg["filter_setup"]
    if not isinstance(fs, dict):
        raise TypeError(f"filter_setup must be a mapping, got {type(fs)}")
    for k in ("dt", "k0", "filter_length"):
        if k not in fs:
            raise ValueError(f"filter_setup missing required key '{k}'")


def validate_model(model: dict) -> None:
    """Fail-fast validation for the canonical in-memory model dict."""
    if not isinstance(model, dict):
        raise TypeError(f"model must be a mapping, got {type(model)}")

    required = [
        "so",
        "sb",
        "area_coef",
        "energy_coef",
        "beta_log10",
        "npow",
        "b0",
        "b1",
        "filter_setup",
        "afilt",
    ]
    missing = [k for k in required if k not in model]
    if missing:
        raise ValueError(f"Model missing required keys: {missing}")

    so = float(model["so"])
    sb = float(model["sb"])
    if not (so > sb > 0.0):
        raise ValueError(f"Require so > sb > 0. Got so={so}, sb={sb}")

    fs = model["filter_setup"]
    if not isinstance(fs, dict):
        raise TypeError(f"filter_setup must be a mapping, got {type(fs)}")
    for k in ("dt", "k0", "filter_length"):
        if k not in fs:
            raise ValueError(f"Model filter_setup missing '{k}'")

    flen = int(fs["filter_length"])
    afilt = model["afilt"]
    if not isinstance(afilt, (list, tuple)):
        raise TypeError(f"afilt must be a list/tuple, got {type(afilt)}")
    if len(afilt) != flen:
        raise ValueError(f"afilt length {len(afilt)} != filter_length {flen}")


def build_model_from_fit(config_yml: str, x_res, coefs) -> dict:
    """Build a canonical model dict from a fit result and fitting config.

    Centralizes fit-space scaling. Nothing else should hard-code constants like
    so/sb or filter setup.
    """
    cfg = parse_config(config_yml)
    validate_fit_config(cfg)
    fs_cfg = cfg["filter_setup"]

    # Interpret x_res using fit_run.outer_params (canonical keys + optional scale).
    fit_run = cfg.get("fit_run") or {}
    outer_params = fit_run.get("outer_params") or []
    if not outer_params:
        raise ValueError("fit_run.outer_params is required to interpret x_res")
    if len(outer_params) != len(x_res):
        raise ValueError(
            f"Length mismatch: outer_params has {len(outer_params)} entries but x_res has {len(x_res)}"
        )
    pvals = {}
    for i, p in enumerate(outer_params):
        key = p.get("key")
        if not key:
            raise ValueError(f"outer_params[{i}] missing 'key'")
        scale = float(p.get("scale", 1.0))
        pvals[key] = float(x_res[i]) * scale

    required_keys = {"beta_log10", "npow", "area_coef", "energy_coef"}
    missing = required_keys - set(pvals)
    if missing:
        raise ValueError(
            f"fit_run.outer_params missing required keys: {sorted(missing)}"
        )

    model = {
        "so": float(cfg["so"]),
        "sb": float(cfg["sb"]),
        "area_coef": float(pvals["area_coef"]),
        "energy_coef": float(pvals["energy_coef"]),
        "beta_log10": float(pvals["beta_log10"]),
        "npow": float(pvals["npow"]),
        "g0": float(cfg.get("g0", 5000.0)),
        "b0": float(coefs["const"]),
        "b1": float(coefs["gnpow"] * 0.001),
        "filter_setup": {
            "dt": str(fs_cfg["dt"]),
            "k0": int(fs_cfg["k0"]),
            "filter_length": int(fs_cfg["filter_length"]),
            "centering": str(fs_cfg.get("centering", "causal")),
        },
        "afilt": [
            float(ak * 1e-3) for ak in coefs[coefs.index.str.startswith("z")].values
        ],
    }
    validate_model(model)
    return model


def parse_config(yml):
    with open(yml) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return config


def read_model_yaml(yml):
    """Read model specification YAML.

    Returns a dict with canonical keys:
      so, sb, area_coef, energy_coef, beta_log10, npow, g0, b0, b1,
      filter_setup {dt, k0, filter_length, centering}, afilt (list[float])
    """
    with open(yml) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    if not isinstance(cfg, dict) or "version" not in cfg:
        raise ValueError(f"Unrecognized model YAML schema in {yml}")

    const = cfg.get("constants", {})
    fs = cfg.get("filter_setup", {})
    gmod = cfg.get("g_model", {})
    phys = cfg.get("physics", {})
    inner = cfg.get("inner_linear", {})

    afilt = fs.get("afilt", cfg.get("afilt", None))
    if afilt is None:
        raise ValueError("Model YAML missing filter coefficients (filter_setup.afilt)")

    out = {
        "version": int(cfg["version"]),
        "so": float(const["so_uS_cm"]),
        "sb": float(const["sb_uS_cm"]),
        "area_coef": float(phys.get("area_coef", 0.0)),
        "energy_coef": float(phys.get("energy_coef", 0.0)),
        "beta_log10": float(gmod["beta_log10"]),
        "npow": float(gmod["npow"]),
        "g0": float(gmod.get("g0", 5000.0)),
        "b0": float(inner["b0"]),
        "b1": float(inner["b1"]),
        "filter_setup": {
            "dt": str(fs["dt"]),
            "k0": int(fs["k0"]),
            "filter_length": int(fs["filter_length"]),
            "centering": str(fs.get("centering", "causal")),
        },
        "afilt": [float(a) for a in afilt],
    }
    validate_model(out)
    return out


def write_model_yaml(model, yml):
    """Write a v1 model specification YAML.

    Parameters
    ----------
    model : dict
        Canonical model dict (as returned by read_model_yaml()).
    yml : str
        Output path.
    """
    validate_model(model)
    fs = model["filter_setup"]
    out = {
        "version": 1,
        "constants": {
            "sb_uS_cm": float(model["sb"]),
            "so_uS_cm": float(model["so"]),
        },
        "filter_setup": {
            "dt": str(fs["dt"]),
            "k0": int(fs["k0"]),
            "filter_length": int(fs["filter_length"]),
            "centering": str(fs.get("centering", "causal")),
            "afilt": [float(a) for a in model["afilt"]],
        },
        "g_model": {
            "beta_log10": float(model["beta_log10"]),
            "npow": float(model["npow"]),
            "g0": float(model.get("g0", 5000.0)),
        },
        "physics": {
            "area_coef": float(model.get("area_coef", 0.0)),
            "energy_coef": float(model.get("energy_coef", 0.0)),
        },
        "inner_linear": {
            "b0": float(model["b0"]),
            "b1": float(model["b1"]),
        },
    }
    with open(yml, "w") as stream:
        yaml.safe_dump(out, stream, sort_keys=False, default_flow_style=False)


def read_fit_run_yaml(yml):
    """Read an optional fit-run YAML (bounds/transforms/solver/window).

    Returns the 'fit_run' dict if present, else {}.
    """
    with open(yml) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    if isinstance(cfg, dict) and "fit_run" in cfg:
        return cfg["fit_run"] or {}
    return {}


def read_fit_yaml(yml):
    with open(yml) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    log10beta = config["log10gbeta"]
    npow = config["npow"]
    area_coef = config["area_coef"]
    energy_coef = config["energy_coef"]
    beta0 = config["b0"]
    beta1 = config["b1"]
    filter_k0 = config["filter_k0"]
    filt_coefs = [an for an in config["afilt"]]
    filter_dt = pd.Timedelta(config["filter_dt"])
    so = config["so"]
    sb = config["sb"]

    return (
        area_coef,
        energy_coef,
        log10beta,
        beta0,
        beta1,
        npow,
        filter_k0,
        filt_coefs,
        filter_dt,
        so,
        sb,
    )


def write_fit_yaml(x_res, coefs, k0, fitting_config, yml):
    """Write the fit results to a yaml file for later use in ec_boundary.

    Parameters
    ----------
    x_res : list or array
        The optimized parameters from the fitting process.
        x_res = [log10beta, npow, area_coef, energy_coef]
    coefs : pandas Series
        The coefficients from the statistical model fit.
        coefs = ['const', 'gnpow', 'z1', 'z2', ...]
    fitting_config : str
        Path to the fitting configuration YAML file.
        yml : str
            Path to the output YAML file to write the fit results.
    """

    cfg = parse_config(fitting_config)
    validate_fit_config(cfg)
    fs_cfg = cfg["filter_setup"]
    so = float(cfg["so"])
    sb = float(cfg["sb"])

    config = {
        "so": float(so),
        "sb": float(sb),
        "area_coef": float(
            x_res[2] * 3600 * 1000000.0
        ),  
        "energy_coef": float(
            x_res[3] * 1000.0
        ),  
        "log10gbeta": float(x_res[0]),  
        "npow": float(x_res[1]),  # x[1] from ec_boundary_fit_gee.py printout
        "b0": float(coefs["const"]),  # from const coef result
        "b1": float(coefs["gnpow"] * 0.001),  # from gnpow coef result
        "afilt": [
            float(ak * 1e-3) for ak in coefs[coefs.index.str.startswith("z")].values
        ],  # z{n} from output coefs
        "filter_k0": int(fs_cfg["k0"]),
        "filter_dt": str(fs_cfg["dt"]),
    }
    with open(yml, "w") as stream:
        yaml.dump(config, stream, default_flow_style=False)
