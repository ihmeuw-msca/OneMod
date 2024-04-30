from onemod.application.ensemble_application import EnsembleApplication
from onemod.application.regmod_smooth_application import RegmodSmoothApplication
from onemod.application.rover_covsel_application import RoverCovselApplication
from onemod.application.weave_application import WeaveApplication


def get_application_class(stage_name: str) -> type:
    # TODO: Complete this map when more applications are implemented
    application_map = {
        "rover_covsel": RoverCovselApplication,
        "regmod_smooth": RegmodSmoothApplication,
        "weave": WeaveApplication,
        "ensemble": EnsembleApplication,
    }
    return application_map[stage_name]
