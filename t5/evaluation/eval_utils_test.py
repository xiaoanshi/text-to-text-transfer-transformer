# Copyright 2020 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for t5.evaluation.eval_utils."""

import collections
import functools
import os
from typing import Callable, Sequence
from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd
from t5.data import dataset_providers
from t5.data import postprocessors
from t5.data import test_utils as data_test_utils
from t5.evaluation import eval_utils
from t5.evaluation import metrics
from t5.evaluation import test_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class EvalUtilsTest(absltest.TestCase):

  def test_parse_events_files(self):
    tb_summary_dir = self.create_tempdir()
    tf.disable_eager_execution()  # Needed in pytest.
    summary_writer = tf.summary.FileWriter(tb_summary_dir.full_path)
    tags = [
        "eval/foo_task/accuracy",
        "eval/foo_task/accuracy",
        "loss",
    ]
    values = [1., 2., 3.]
    steps = [20, 30, 40]
    for tag, value, step in zip(tags, values, steps):
      summary = tf.Summary()
      summary.value.add(tag=tag, simple_value=value)
      summary_writer.add_summary(summary, step)
    summary_writer.flush()
    events = eval_utils.parse_events_files(tb_summary_dir.full_path)
    self.assertDictEqual(
        events,
        {
            "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
            "loss": [(40, 3.)],
        },
    )

  def test_get_eval_metric_values(self):
    events = {
        "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
        "eval/bar_task/sequence_accuracy": [(10, 3.)],
        "loss": [(40, 3.)],
    }
    eval_values = eval_utils.get_eval_metric_values(events)
    self.assertDictEqual(
        eval_values,
        {
            "foo_task/accuracy": [(20, 1.), (30, 2.)],
            "bar_task/sequence_accuracy": [(10, 3.)],
        }
    )

  def test_glue_average(self):
    mets = eval_utils.METRIC_NAMES.items()
    glue_metric_names = [
        v.name for k, v in mets if k.startswith("glue") and "average" not in k
    ]
    super_glue_metric_names = [
        v.name for k, v in mets if k.startswith("super") and "average" not in k
    ]
    extra_metric_names = ["Fake metric", "Average GLUE Score"]
    columns = glue_metric_names + super_glue_metric_names + extra_metric_names
    n_total_metrics = len(columns)
    df = pd.DataFrame(
        [np.arange(n_total_metrics), 2*np.arange(n_total_metrics)],
        columns=columns,
    )
    df = eval_utils.compute_avg_glue(df)
    expected_glue = (
        0 + 1 + (2 + 3)/2. + (4 + 5)/2. + (6 + 7)/2. + (8 + 9)/2. + 10 + 11
    )/8.
    self.assertSequenceAlmostEqual(
        df["Average GLUE Score"], [expected_glue, 2*expected_glue]
    )
    expected_super = (
        12 + (13 + 14)/2. + 15 + (16 + 17)/2. + (18 + 19)/2. + 20 + 21 + 22
    )/8.
    self.assertSequenceAlmostEqual(
        df["Average SuperGLUE Score"], [expected_super, 2*expected_super]
    )
    del df["CoLA"]
    del df["Average GLUE Score"]
    df = eval_utils.compute_avg_glue(df)
    self.assertNoCommonElements(df.columns, ["Average GLUE Score"])

  def test_metric_group_max(self):
    df = pd.DataFrame(
        collections.OrderedDict([
            ("ABC Accuracy", [1., 2., 3., 4.]),
            ("DEF Exact Match", [0., 10., 3., 0.]),
            ("DEF Accuracy", [4., 7., 8., 0.]),
        ]),
        index=[10, 20, 30, 40],
    )
    metric_names = collections.OrderedDict([
        ("metric1", eval_utils.Metric("ABC Accuracy")),
        ("metric2", eval_utils.Metric("DEF Accuracy", "DEF")),
        ("metric3", eval_utils.Metric("DEF Exact Match", "DEF")),
    ])
    metric_max, metric_max_step = eval_utils.metric_group_max(df, metric_names)
    self.assertTrue(metric_max.keys().equals(df.columns))
    self.assertListEqual(list(metric_max.values), [4., 10., 7.])
    self.assertTrue(metric_max_step.keys().equals(df.columns))
    self.assertSequenceEqual(list(metric_max_step.values), [40, 20, 20])

  def test_log_csv(self):
    metric_names = list(eval_utils.METRIC_NAMES.values())
    df = pd.DataFrame(
        collections.OrderedDict([
            (metric_names[0].name, [np.nan, 1., 2.]),
            (metric_names[1].name, [3., np.nan, np.nan]),
            (metric_names[2].name, [4., np.nan, np.nan]),
        ]),
        index=[10, 20, 30],
    )
    df.index.name = "step"
    output_file = os.path.join(self.create_tempdir().full_path, "results.csv")
    eval_utils.log_csv(df, output_file=output_file)
    with tf.io.gfile.GFile(output_file) as f:
      output = f.read()
    expected = """step,{},{},{}
10,,3.000,4.000
20,1.000,,
30,2.000,,
max,2.000,3.000,4.000
step,30,10,10""".format(*[m.name for m in metric_names[:3]])
    self.assertEqual(output, expected)


def register_dummy_task(
    task_name: str,
    dataset_fn: Callable[[str, str], tf.data.Dataset],
    output_feature_names: Sequence[str] = ("inputs", "targets"),
    postprocess_fn=None,
    metrics_fn=None) -> None:
  """Register a dummy task for GetDatasetTest."""
  dataset_providers.TaskRegistry.add(
      task_name,
      dataset_providers.TaskV3,
      source=dataset_providers.FunctionSource(
          dataset_fn=dataset_fn, splits=["train", "validation"]),
      preprocessors=[dataset_providers.CacheDatasetPlaceholder()],
      postprocess_fn=postprocess_fn,
      output_features={
          feat: dataset_providers.Feature(data_test_utils.sentencepiece_vocab())
          for feat in output_feature_names
      },
      metric_fns=metrics_fn)


class EvaluatorTest(test_utils.BaseMetricsTest):

  def test_evaluate_single_task(self):
    task = mock.Mock()
    task.name = "mocked_task"
    task.metric_fns = [metrics.sequence_accuracy]
    # Identity postprocess function
    task.postprocess_fn = lambda d, example, is_target: d

    def mock_init(self):
      # dummy prediction function which always returns the same output
      self._predict_fn = lambda x: ["example 1", "example 2", "example 3"]
      self._cached_ds = {task.name: None}
      # cached_examples are dummy values.
      self._cached_examples = {
          task.name: [{"targets": 1}, {"targets": 1}, {"targets": 1}]
      }
      self._cached_targets = {
          task.name: ["example 1", "example 1", "example 3"]
      }
      self._eval_tasks = [task]

    with mock.patch.object(eval_utils.Evaluator, "__init__", new=mock_init):
      evaluator = eval_utils.Evaluator()
      eval_results = evaluator.evaluate(compute_metrics=True)
      expected = {task.name: [{"sequence_accuracy": 2.0 / 3 * 100}]}
      self.assertDictClose(expected[task.name][0], eval_results[task.name][0])

  def test_evaluate_single_task_with_postprocessor(self):
    task = mock.Mock()
    task.name = "mocked_task"
    task.metric_fns = [metrics.accuracy]
    # Identity postprocess function
    task.postprocess_fn = functools.partial(
        postprocessors.string_label_to_class_id, label_classes=["1", "2", "3"])

    def mock_init(self):
      # dummy prediction function which always returns the same output
      self._predict_fn = lambda x: ["2", "2", "3"]  # label = [1, 1, 2]
      self._cached_ds = {task.name: None}
      # cached_examples are dummy values.
      self._cached_examples = {
          task.name: [{"targets": 1}, {"targets": 1}, {"targets": 1}]
      }
      self._cached_targets = {task.name: [0, 1, 2]}
      self._eval_tasks = [task]

    with mock.patch.object(eval_utils.Evaluator, "__init__", new=mock_init):
      evaluator = eval_utils.Evaluator()
      eval_results = evaluator.evaluate(compute_metrics=True)
      expected = {task.name: [{"accuracy": 2.0 / 3 * 100}]}
      self.assertDictClose(expected[task.name][0], eval_results[task.name][0])

  def test_evaluate_mixture(self):
    task1 = mock.Mock()
    task1.name = "mocked_task1"
    task1.metric_fns = [metrics.sequence_accuracy]
    # Identity postprocess function
    task1.postprocess_fn = lambda d, example, is_target: d

    task2 = mock.Mock()
    task2.name = "mocked_task2"
    task2.metric_fns = [metrics.accuracy]
    # Identity postprocess function
    task2.postprocess_fn = lambda d, example, is_target: d

    def mock_init(self):
      # dummy prediction functions which always return the same output
      self._cached_ds = {task1.name: mock.Mock(), task2.name: mock.Mock()}

      def predict_fn(ds):
        if ds == self._cached_ds[task1.name]:
          return ["example 1", "example 2", "example 3"]
        elif ds == self._cached_ds[task2.name]:
          return [0, 2, 1]

      self._predict_fn = predict_fn
      self._cached_examples = {
          task1.name: [{"targets": 1}, {"targets": 1}, {"targets": 1}],
          task2.name: [{"targets": 1}, {"targets": 1}, {"targets": 1}]
      }
      self._cached_targets = {
          task1.name: ["example 1", "example 1", "example 3"],
          task2.name: [0, 0, 2]
      }
      self._eval_tasks = [task1, task2]

    with mock.patch.object(eval_utils.Evaluator, "__init__", new=mock_init):
      evaluator = eval_utils.Evaluator()
      eval_results = evaluator.evaluate(compute_metrics=True)
      expected = {
          task1.name: [{"sequence_accuracy": 2.0 / 3 * 100}],
          task2.name: [{"accuracy": 1.0 / 3 * 100}]
      }
      self.assertDictClose(expected[task1.name][0], eval_results[task1.name][0])
      self.assertDictClose(expected[task2.name][0], eval_results[task2.name][0])


if __name__ == "__main__":
  absltest.main()
