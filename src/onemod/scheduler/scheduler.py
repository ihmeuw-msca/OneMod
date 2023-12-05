from typing import Generator

from onemod.actions.action import Action


class Scheduler:

    def __init__(self, stages: list[str]):
        self.stages = stages

    def parent_action_generator(self) -> Generator[Action, None, None]:
        for stage in self.stages:
            application = get_application(stage)  # TODO: A simple lookup table should suffice
            generator = application.action_generator()
            yield from generator

    def run(self, run_local: bool = False):
        if run_local:
            for action in self.parent_action_generator():
                action.evaluate()
        else:
            workflow = self.create_workflow()
            tasks = [action.task for action in self.parent_action_generator()]
            workflow.add_tasks(tasks)
            workflow.run(configure_logging=True)
