# Configurations

`experiment.yaml` is the operational CT baseline; `smoke.yaml` uses tiny synthetic
CPU inputs. Load with `pulmoai.config.load_config`, or validate with
`python -m pulmoai validate-config --config smoke`.

`legacy_cxr.yaml` retains the original six CXR groups for
`respiratory_xray.config.load_config`. Those are research settings without an
implemented CXR model or trainer. They must not be passed to the CT CLI.

The existing YAML composer is retained. Local paths are CLI arguments. No `.env`
or credentials are loaded. `local.example.yaml` documents the older optional
explicit settings helper; `local.yaml` is ignored and never automatically merged.
