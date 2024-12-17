"""Run pipeline through Jobmon on dummy cluster.

Since dummy cluster doesn't actually evaluate tasks, this test just
makes sure the workflow finishes successfully.

"""

import pytest
from tests.helpers.jobmon_helpers import setup_pipeline


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["run", "fit", "predict"])
def test_pipeline(tmp_path, method):
    # Setup the pipeline
    pipeline = setup_pipeline(tmp_path)

    # Run pipeline through Jobmon on dummy cluster
    pipeline.evaluate(
        method=method,
        backend="jobmon",
        cluster="dummy",
        resources={"tool_resources": {"dummy": {"queue": "null.q"}}},
    )
