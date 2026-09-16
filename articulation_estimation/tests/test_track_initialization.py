"""CPU regression tests: run python -m unittest discover -s articulation_estimation/tests."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from articulation_estimation.cli import parse_args
from articulation_estimation.fitting import fit_joint_hypotheses
from articulation_estimation.geometry import so3_exp_np
from articulation_estimation.tracks import PointTrackSequence, rigid_pose_from_tracks


class TrackInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.label_root = self.root / 'trackcraft' / 'moving'
        (self.label_root / 'scene_flow').mkdir(parents=True)
        self.rng = np.random.default_rng(11)
        self.points = self.rng.normal(size=(150, 3)) * 0.2 + [0.2, -0.1, 2.0]
        self.axis = np.array([0.2, 0.8, 0.3])
        self.axis /= np.linalg.norm(self.axis)
        self.pivot = np.array([-0.4, 0.1, 2.2])

    def write_tracks(self, kind='revolute', frames=range(1, 9)):
        np.save(self.label_root / 'pts3d_ref.npy', self.points)
        (self.root / 'trackcraft' / 'config.json').write_text(json.dumps({'ref_frame': 1}))
        for frame in frames:
            if frame == 1:
                continue
            q = (frame - 1) * 0.09
            target = (self.points + q * self.axis if kind == 'prismatic' else
                      (self.points - self.pivot) @ so3_exp_np(q * self.axis).T + self.pivot)
            np.save(self.label_root / 'scene_flow' / f'{frame:06d}.npy', target - self.points)

    def test_rigid_registration_with_outliers_and_invalid_points(self):
        rotation = so3_exp_np(self.axis * 0.5)
        translation = np.array([0.1, -0.3, 0.2])
        target = self.points @ rotation.T + translation
        target[:15] += self.rng.normal(size=(15, 3)) * 2.0
        target[15:18] = np.nan
        pose = rigid_pose_from_tracks(self.points, target)
        np.testing.assert_allclose(pose.transform[:3, :3], rotation, atol=1e-8)
        np.testing.assert_allclose(pose.transform[:3, 3], translation, atol=1e-8)
        self.assertEqual(pose.valid_points, 147)
        self.assertGreater(pose.confidence, 0.9)

    def test_both_joint_types_with_different_track_and_mesh_references(self):
        for kind in ['prismatic', 'revolute']:
            with self.subTest(kind=kind):
                self.write_tracks(kind)
                sequence = PointTrackSequence(self.root, 'moving', reference_frame=4)
                frames = [1, 2, 4, 6, 8]
                poses = sequence.estimate_poses(frames)
                fit = fit_joint_hypotheses(
                    np.stack([poses[t].transform for t in frames]), sequence.centroid,
                    sequence.diameter, reference_index=2, ransac_iterations=8)[kind]
                self.assertGreater(abs(fit.axis @ self.axis), 1 - 1e-7)
                sign = np.sign(fit.axis @ self.axis)
                np.testing.assert_allclose(fit.states, sign * (np.array(frames) - 4) * 0.09,
                                           atol=1e-6)
                if kind == 'revolute':
                    self.assertLess(np.linalg.norm(np.cross(fit.pivot - self.pivot, self.axis)), 1e-6)
                self.assertLess(fit.residual, 1e-6)
                np.testing.assert_array_equal(poses[4].transform, np.eye(4))

    def test_stage40_frame_names_are_not_camera_record_indices(self):
        self.write_tracks('prismatic')
        sequence = PointTrackSequence(self.root, 'moving', 1)
        np.testing.assert_allclose(sequence.positions(3), self.points + 0.18 * self.axis)
        np.testing.assert_allclose(sequence.positions(2), self.points + 0.09 * self.axis)

    def test_missing_metadata_or_reference_coverage_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, 'config.json'):
            PointTrackSequence(self.root, 'moving', 1)
        self.write_tracks()
        with self.assertRaisesRegex(FileNotFoundError, 'no 3D tracks for frame 20'):
            PointTrackSequence(self.root, 'moving', 20)

    def test_malformed_flow_cannot_broadcast(self):
        self.write_tracks()
        np.save(self.label_root / 'scene_flow' / '000002.npy', np.zeros((1, 3)))
        sequence = PointTrackSequence(self.root, 'moving', 1)
        with self.assertRaisesRegex(ValueError, 'expected'):
            sequence.estimate_poses([1, 2])

    def test_invalid_and_degenerate_correspondences(self):
        with self.assertRaisesRegex(ValueError, 'finite tracks'):
            rigid_pose_from_tracks(self.points, np.full_like(self.points, np.nan))
        line = np.column_stack([np.arange(8), np.zeros((8, 2))])
        with self.assertRaisesRegex(ValueError, 'collinear'):
            rigid_pose_from_tracks(line, line + 1)

    def test_cli_default_video_ablation_and_compatibility_alias(self):
        self.assertEqual(parse_args(['--scene-dir', '.']).joint_initializer, 'tracks')
        self.assertEqual(parse_args(['--scene-dir', '.', '--joint-initializer', 'video']).joint_initializer, 'video')
        self.assertEqual(parse_args(['--scene-dir', '.', '--use-track-prior']).joint_initializer, 'tracks')

    def test_default_pipeline_skips_image_pose_tracking_and_keeps_validation_out(self):
        import torch
        from articulation_estimation import pipeline as p
        self.write_tracks('prismatic')
        mask = np.ones((10, 10), dtype=bool)
        part = SimpleNamespace(vertices=self.points.astype(np.float32), faces=np.array([[0, 1, 2]]))
        # Dense video extends beyond the track window; only initialization is restricted.
        scene = SimpleNamespace(
            root=self.root, reference_frame=4, static_label='static', index_offset=1,
            pose_initializer_kind='sam3d', pose_source='sam3d/pose.json',
            parts={'moving': part, 'static': part}, paired_frames=lambda label: list(range(1, 13)),
            image=lambda frame: np.zeros((10, 10, 3), dtype=np.uint8),
            mask=lambda label, frame: mask)
        refined = []
        held_out = []
        def refine(fit, frames, scene, label, renderer, moving_mesh, static_mesh, maps, config):
            refined.extend(frames)
            return p.RefinedHypothesis(fit.joint_type, fit.axis, fit.pivot,
                                      dict(zip(frames, fit.states)), moving_mesh[0], static_mesh[0], 0.0)
        def validate(hyp, frames, *args):
            held_out.extend(frames)
            hyp.validation_loss = 0.1 if hyp.joint_type == 'prismatic' else 1.0
        def mesh(scene, renderer, part, *args):
            return (torch.tensor(part.vertices), torch.tensor(part.faces), None, None)
        with patch.object(p, 'DifferentiableMeshRenderer'), \
             patch.object(p, 'optimize_static_pose', return_value=(scene.parts, {'loss': 0.0})), \
             patch.object(p, 'prepare_features', return_value=(None, {t: None for t in range(1, 13)})), \
             patch.object(p, 'make_feature_mesh', side_effect=mesh), \
             patch.object(p, 'track_sparse_poses', side_effect=AssertionError('image pose tracker called')), \
             patch.object(p, 'refine_hypothesis', side_effect=refine), \
             patch.object(p, 'validate_hypothesis', side_effect=validate), \
             patch.object(p, 'recover_dense_states', side_effect=lambda h, frames, *a: {t: (t - 4) * .09 for t in frames}):
            result = p.estimate_label(scene, 'moving', p.PipelineConfig(ransac_iterations=4), progress=lambda x: None)
        self.assertEqual(result['joint_initializer'], 'tracks')
        self.assertEqual(result['type'], 'prismatic')
        self.assertEqual(result['track_initialization']['track_reference_frame'], 1)
        self.assertEqual(result['track_initialization']['mesh_reference_frame'], 4)
        self.assertTrue(held_out)
        self.assertFalse(set(refined) & set(held_out))
        self.assertEqual(set(result['track_initialization']['fit_frames']), set(refined))
        self.assertEqual(len(result['joint_states']), 12)
        self.assertFalse(result['inputs']['tracks_used_in_refinement'])
        self.assertIn('initial_joint', result['model_candidates']['prismatic'])
        json.dumps({k: v for k, v in result.items() if not k.startswith('_')})

        # The explicit video ablation still works without consulting track files.
        def video_poses(scene, label, renderer, mesh, frames, maps, config):
            poses = {}
            for frame in frames:
                transform = np.eye(4)
                transform[:3, 3] = self.axis * ((frame - 4) * .09)
                poses[frame] = p.TrackedPose(frame, transform, np.zeros(3),
                                             transform[:3, 3], 0.)
            return poses
        with patch.object(p, 'PointTrackSequence', side_effect=AssertionError('tracks loaded in video mode')), \
             patch.object(p, 'DifferentiableMeshRenderer'), \
             patch.object(p, 'optimize_static_pose', return_value=(scene.parts, {'loss': 0.0})), \
             patch.object(p, 'prepare_features', return_value=(None, {t: None for t in range(1, 13)})), \
             patch.object(p, 'make_feature_mesh', side_effect=mesh), \
             patch.object(p, 'track_sparse_poses', side_effect=video_poses) as video_tracker, \
             patch.object(p, 'refine_hypothesis', side_effect=refine), \
             patch.object(p, 'validate_hypothesis', side_effect=validate), \
             patch.object(p, 'recover_dense_states', side_effect=lambda h, frames, *a: {t: (t - 4) * .09 for t in frames}):
            result = p.estimate_label(scene, 'moving', p.PipelineConfig(
                joint_initializer='video', ransac_iterations=4), progress=lambda x: None)
        video_tracker.assert_called_once()
        self.assertEqual(result['joint_initializer'], 'video')
        self.assertIsNone(result['track_initialization'])
        self.assertFalse(result['inputs']['track_prior_enabled'])


    def test_joint_parameters_improve_using_only_image_loss(self):
        import torch
        from articulation_estimation import pipeline as p
        from articulation_estimation.fitting import JointFit
        from articulation_estimation.rendering import Observation, image_loss
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-.5, .5, 28),
                                       torch.linspace(-.5, 1.1, 42), indexing='ij')
        class AnalyticRenderer:
            # CPU differentiable silhouette renderer isolates the image refinement
            # from nvdiffrast/CUDA while exercising the real objective and optimizer.
            device = torch.device('cpu')
            def render(self, vertices, faces, camera, *args, **kwargs):
                center = vertices.mean(dim=0)
                alpha = (torch.sigmoid((.16 - (grid_x - center[0]).abs()) * 100)
                         * torch.sigmoid((.12 - (grid_y - center[1]).abs()) * 100))
                return {'alpha': alpha}
        renderer = AnalyticRenderer()
        vertices = torch.tensor([[-.1, -.1, 2.], [.1, -.1, 2.], [0., .2, 2.]])
        faces = torch.tensor([[0, 1, 2]])
        mesh = (vertices, faces, None, None)
        frames = [1, 2, 3]
        truth = [0., .3, .6]
        observations = {}
        for frame, q in zip(frames, truth):
            mask = (renderer.render(vertices + torch.tensor([q, 0., 0.]), faces, None)['alpha'] > .5).float()
            observations[frame] = Observation(frame, mask, torch.zeros_like(mask), None)
        axis = np.array([1., .25, 0.])
        axis /= np.linalg.norm(axis)
        initial_states = np.array([0., .2, .45])
        fit = JointFit('prismatic', axis.copy(), None, initial_states, 0.,
                       np.ones(3, dtype=bool), np.zeros(3))
        config = p.PipelineConfig(device='cpu', use_features=False, state_iters=30,
                                  joint_iters=180, global_iters=0, lambda_smooth=0.,
                                  lambda_shape=0., lambda_boundary=0.)
        def evaluate(current_axis, states):
            losses = []
            for frame, q in zip(frames, states):
                moved = vertices + float(q) * torch.as_tensor(current_axis, dtype=torch.float32)
                loss, _ = image_loss(renderer.render(moved, faces, None),
                                     observations[frame], p._weights(config, False))
                losses.append(float(loss))
            return np.mean(losses)
        before = evaluate(axis, initial_states)
        scene = SimpleNamespace(reference_frame=1, static_label='static')
        with patch.object(p, '_observation', side_effect=lambda scene, label, frame, *a, **kw: observations[frame]):
            refined = p.refine_hypothesis(fit, frames, scene, 'moving', renderer,
                                          mesh, mesh, {}, config)
        after = evaluate(refined.axis, [refined.states[t] for t in frames])
        self.assertLess(after, before * .75)
        self.assertLess(abs(refined.axis[1]), abs(axis[1]))
        self.assertEqual(refined.states[1], 0.)

if __name__ == '__main__':
    unittest.main()
