# Utilities for the library

import yaml
import pandas as pd


def parse_config(yml):
    with open(yml) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return config


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
    filt_coefs = [an * 1e-3 for an in config["afilt"]]
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

    # filt_coefs = [ak * 1e-3 for ak in coefs[coefs.index.str.startswith("z")].values]
    so = 20000.0  # hardwired in ec_boundary_fit_gee.py
    sb = 200.0  # hardwired in ec_boundary_fit_gee.py
    # filt_coefs = np.array(params["afilt"]) * 1e-3  # Convert to numpy array and scale

    config = {
        "so": float(so),
        "sb": float(sb),
        "area_coef": float(
            x_res[2] * 3600 * 1000000.0
        ),  # x[2] from ec_boundary_fit_gee.py printout
        "energy_coef": float(
            x_res[3] * 1000.0
        ),  # x[3] from ec_boundary_fit_gee.py printout
        "log10gbeta": float(x_res[0]),  # x[0] from ec_boundary_fit_gee.py printout
        "npow": float(x_res[1]),  # x[1] from ec_boundary_fit_gee.py printout
        "b0": float(coefs["const"]),  # from const coef result
        "b1": float(coefs["gnpow"] * 0.001),  # from gnpow coef result
        "afilt": [
            float(ak * 1e3) for ak in coefs[coefs.index.str.startswith("z")].values
        ],  # z{n} from output coefs
        "filter_k0": int(k0),  # from fitting_config yaml file
        "filter_dt": "3h",  # from fitting_config.yaml
    }
    with open(yml, "w") as stream:
        yaml.dump(config, stream, default_flow_style=False)
