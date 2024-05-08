from onemod.scheduler.scheduling_utils import ParentTool, TaskTemplateFactory


def test_task_template_factory(testing_tool):
    """Check that given actions, we get the right task templates."""

    ParentTool.tool = testing_tool

    task_template = TaskTemplateFactory.get_task_template(
        action_name="initialize_results",
        resources_path="",
    )
    assert task_template.template_name == "initialize_results"
    assert (
        task_template.active_task_template_version.command_template
        == "{entrypoint} --stages {stages} --directory {directory}"
    )

    task_template = TaskTemplateFactory.get_task_template(
        action_name="collect_results",
        resources_path="",
    )
    assert task_template.template_name == "collect_results"
    assert (
        task_template.active_task_template_version.command_template
        == "{entrypoint} --stage_name {stage_name} --directory {directory}"
    )

    task_template = TaskTemplateFactory.get_task_template(
        action_name="rover_covsel_model",
        resources_path="",
    )
    assert task_template.template_name == "rover_covsel_model"
    assert (
        task_template.active_task_template_version.command_template
        == "{entrypoint} --submodel_id {submodel_id} --directory {directory}"
    )

    task_template = TaskTemplateFactory.get_task_template(
        action_name="spxmod_model",
        resources_path="",
    )
    assert task_template.template_name == "spxmod_model"
    assert (
        task_template.active_task_template_version.command_template
        == "{entrypoint} --directory {directory}"
    )
