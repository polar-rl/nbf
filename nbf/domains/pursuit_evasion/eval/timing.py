from nbf.domains.pursuit_evasion.eval.dispatch import PE_EXTRA_SAMPLERS
from nbf.eval.timing import time_method_walltime as _time_method_walltime


def time_method_walltime(spec: str, **kwargs) -> dict[str, float]:
    return _time_method_walltime(spec, extra_samplers=PE_EXTRA_SAMPLERS, **kwargs)
