"""Exercise real dispatch/database with a deterministic embedding adapter."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import cv2
import numpy as np

from test_face_candidates import load_face

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.face_corpus import corpus_entries, load_image


class RegistrationContracts(unittest.TestCase):
    def setUp(self):
        self.ns = load_face()
        self.ns.update(threading=threading, Path=Path, json=json,
                       corpus_entries=corpus_entries, load_image=load_image)
        self.db = self.ns['FaceDatabase']()
        self.plugin = self.ns['FaceRecognitionPlugin'].__new__(self.ns['FaceRecognitionPlugin'])
        self.plugin._face_db = self.db
        self.plugin._model_lock = threading.Lock()
        self.plugin._similarity_threshold = .4
        self.plugin._max_batch = 1000
        class Adapter:
            def detect_and_embed(self, image):
                if not image.any():
                    return []
                embedding = np.array([1., 0.]) if image.mean() < 150 else np.array([0., 1.])
                return [dict(embedding=embedding, bbox=[0, 0, 8, 8], confidence=.9)]
        self.plugin._model = Adapter()

    def call(self, action, **args):
        return self.plugin.dispatch('face', dict(action=action, **args))

    def test_corpus_lifecycle_and_shared_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('a.png', 80), ('b.png', 80), ('c.png', 220), ('bad.png', 0)]:
                cv2.imwrite(str(root / name), np.full((10, 10, 3), value, np.uint8))
            (root / 'manifest.json').write_text(json.dumps([
                dict(file='a.png', name='Alice', person='dataset-a'),
                dict(file='b.png', name='Alice', person='dataset-a'),
                dict(file='c.png', name='Bob', person='dataset-b')]))
            result = self.call('register_by_corpus', package=directory)
            self.assertTrue(result['ok'], result)
            self.assertEqual((result['total'], result['registered'], result['failed']), (4, 3, 1))
            rows = {row['file']: row for row in result['results']}
            pid = rows['a.png']['person_id']
            self.assertEqual(pid, rows['b.png']['person_id'])
            self.assertNotEqual(pid, rows['c.png']['person_id'])
            self.assertNotEqual(pid, 'dataset-a')
            self.assertEqual(self.db.match(np.array([1., 0.]), .4)[0], pid)
            self.assertEqual(self.call('get_person', person_id=pid)['person']['samples'], 2)
            update = self.call('update_person', person_id=pid, name='Alice Updated', profile={'role': 'test'})
            self.assertEqual(update['person']['name'], 'Alice Updated')
            photo = self.call('recognize_by_photo', image_path=str(root / 'a.png'))
            self.assertEqual(photo['faces'][0]['identity']['person_id'], pid)
            self.assertEqual(photo['faces'][0]['name'], 'Alice Updated')
            count = self.call('list_persons')['total']
            self.call('recognize_by_photo', image_path=str(root / 'a.png'))
            self.assertEqual(self.call('list_persons')['total'], count)
            deletion = self.call('forget', person_id=pid)
            self.assertTrue(deletion['ok'], deletion)
            self.assertEqual(self.db.match(np.array([1., 0.]), .4)[0], 'unknown')

    def test_zip_dispatch(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'batch.zip'
            _, image = cv2.imencode('.png', np.full((10, 10, 3), 80, np.uint8))
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr('alice.png', image.tobytes())
                archive.writestr('manifest.json', json.dumps([
                    {'file': 'alice.png', 'name': 'Alice', 'person': 'benchmark-id'}]))
            result = self.call('register_by_corpus', package=str(package))
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['registered'], 1)
            pid = result['results'][0]['person_id']
            face = self.ns['_face_result'](self.db, dict(embedding=np.array([1., 0.]),
                                          bbox=[0, 0, 8, 8], confidence=.9), (10, 10, 3), .4)
            self.assertEqual(face['person_id'], pid)
            self.assertEqual(face['identity']['person_id'], pid)
            self.assertTrue(face['known'])
            self.assertEqual(face['bbox_relative'], [0., 0., .8, .8])

    def test_invalid_requests(self):
        for action, args in [('register_by_corpus', {}), ('get_person', {'person_id': 'absent'}),
                             ('register_by_photo', {'image_path': '/does/not/exist'})]:
            self.assertFalse(self.call(action, **args)['ok'])

    def test_concurrent_enrollment(self):
        threads = [threading.Thread(target=self.db.enroll, args=(np.array([1., 0.]),),
                                    kwargs={'name': 'Alice'}) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        roster = self.db.list_persons()
        self.assertEqual(roster['total'], 1)
        self.assertEqual(roster['persons'][0]['samples'], 8)


if __name__ == '__main__':
    unittest.main()
