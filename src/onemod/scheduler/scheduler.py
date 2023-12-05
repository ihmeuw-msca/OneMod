
class Scheduler:

    def __init__(self, stages: list[str], run_local: bool = True):
        self.run_local = run_local
        self.stages = stages
        self.tool = None
        if not self.run_local:
            self.create_tool()

    def run(self):
        for stage in self.stages:
            application = get_application(stage)
            if self.run_local:
                tasks = list(application.action_generator(self.run_local))
            else:
                run_tasks(application.action_generator(self.run_local))