from onemod.application.rover_covsel_application import RoverCovselApplication
from onemod.application.regmod_smooth_application import RegmodSmoothApplication


def get_application_class(stage_name: str):
    # TODO: Complete this map when more applications are implemented
    application_map = {
        'rover_covsel': RoverCovselApplication,
        'regmod_smooth': RegmodSmoothApplication,
    }

    return application_map[stage_name]