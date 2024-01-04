from onemod.application.rover_covsel_application import RoverCovselApplication


def get_application_class(stage_name: str):
    # TODO: Complete this map when more applications are implemented
    application_map = {
        'rover_covsel': RoverCovselApplication,
    }

    return application_map[stage_name]